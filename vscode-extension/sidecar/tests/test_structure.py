"""Questions about how the code fits together: plain words vs names written as code, folders as
starting points, the graph's relationships in the prompt, and structural questions answered
from the graph (structural.py)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.dirname(HERE)
sys.path[:0] = [HERE, SIDECAR, os.path.abspath(os.path.join(SIDECAR, "..", "..", "core"))]

from graphrag_code import retrieve as R  # noqa: E402
from graphrag_code import structural as S  # noqa: E402
from graphrag_code.embed import HashingEmbedder  # noqa: E402
from graphrag_code.graph import looks_like_code  # noqa: E402
from graphrag_code.index import Index  # noqa: E402
from graphrag_code.workflows import ASK_SYSTEM, Workflows  # noqa: E402
from test_sidecar import FILES, write_repo  # noqa: E402

CLI = {
    "cli/__init__.py": "",
    "cli/run.py": '''
        from pkg.service import shout
        from pkg.models import format_name


        def main(name):
            return shout(name) + format_name(name)
        ''',
}
M, SV, C = "pkg/models.py::", "pkg/service.py::", "cli/run.py::"


class Fake:
    """A model that records its prompts, gives a fixed answer, and replies to graph-tool steps
    from a script (then "done")."""

    def __init__(self, reply="ok", commands=()):
        self.reply, self.prompts, self.commands, self.tool_prompts = reply, [], list(commands), []

    def stream(self, messages):
        self.prompts.append(messages)
        yield self.reply

    def invoke(self, messages):
        self.tool_prompts.append(messages)
        return type("Reply", (), {"content": self.commands.pop(0) if self.commands else "done"})()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = os.path.join(self.tmp.name, "repo")
        write_repo(root)
        write_repo(root, CLI)
        self.index = Index(root, os.path.join(self.tmp.name, "s", "i.sqlite"), HashingEmbedder(), vector_store="sqlite")
        self.index.build()
        self.g = self.index.graph

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()


class SeedsTest(Base):
    def test_names_written_as_code_outweigh_plain_words(self):
        self.assertTrue(looks_like_code("format_name", ""))
        self.assertTrue(looks_like_code("Registry", ""))
        self.assertTrue(looks_like_code("greet", "what does `greet` return?"))
        self.assertFalse(looks_like_code("greet", "how do users greet each other?"))
        self.assertEqual(self.g.seed_candidates("how does format_name work?"), {M + "format_name": 1.0})
        self.assertEqual(self.g.seed_candidates("how do users greet each other?"), {M + "User.greet": 0.3})

    def test_a_plain_word_is_a_hint_not_what_the_question_is_about(self):
        r = R.retrieve(self.index, "how do users greet each other?")
        self.assertEqual(r["about"], [])

    def test_folders(self):
        self.assertEqual(set(self.g.named_folders("does cli use pkg?")), {"cli", "pkg"})
        self.assertEqual(self.g.named_folders("what is in the app folder? the src?"), {})  # generic names


class FolderNodesTest(Base):
    def test_folders_are_nodes_but_never_code(self):
        self.assertEqual(self.g.symbols["pkg/::"].kind, "folder")
        self.assertTrue([e for e in self.g.out["pkg/::"] if e.dst == "pkg/models.py::" and e.kind == "contains_dir"])
        self.assertEqual(self.index.graph.stats()["folders"], 2)
        self.assertNotIn("folder", self.index.graph.stats()["symbols"])
        self.assertFalse([s for s in self.index.vector_search("pkg models service", k=50) if s.endswith("/::")])
        r = R.retrieve(self.index, "does cli use pkg?")
        self.assertGreater(r["ppr"].get("pkg/::", 0), 0)  # the named folder's node is a seed
        self.assertFalse([c for c in r["context"] if c["kind"] == "folder"])
        self.assertFalse([x for x in r["relations"] if x["kind"] == "contains_dir"])


class RelationshipsTest(Base):
    def test_two_folders_get_their_links_and_connecting_code(self):
        r = R.retrieve(self.index, "is anything in pkg called by the cli?")
        self.assertEqual(set(r["folders"]), {"cli", "pkg"})
        links = {(l["from"], l["to"]): l for l in r["folderLinks"]}
        self.assertEqual(links[("cli", "pkg")]["count"], 4)  # main calls shout and format_name; the module imports both
        self.assertEqual(links[("pkg", "cli")]["count"], 0)
        texts = [x["text"] for x in r["relations"]]
        self.assertIn("cli/run.py::main --calls--> pkg/service.py::shout", texts)
        ids = [c["id"] for c in r["context"]]
        self.assertIn(C + "main", ids)  # the code that connects them is read
        self.assertTrue(any(i.startswith("pkg/") for i in ids))

    def test_the_prompt_has_the_relationships_and_no_false_no(self):
        llm = Fake()
        Workflows(self.index, llm).ask({"query": "how does the cli produce its output?"}, lambda e: None)
        prompt = llm.prompts[0][1]["content"]
        self.assertIn("Relationships (code graph, static analysis):", prompt)
        self.assertIn("--calls-->", prompt)
        self.assertIn("Never conclude that something doesn't happen", ASK_SYSTEM)


class StructuralTest(Base):
    def kind_and_targets(self, q, focus=None):
        c = S.classify(self.index, q, focus)
        return (c["kind"], [t.label for t in c["targets"]]) if c else None

    def test_classification(self):
        self.assertEqual(self.kind_and_targets("who calls format_name?"), ("callers", ["format_name"]))
        self.assertEqual(self.kind_and_targets("where is `shout` used?"), ("callers", ["shout"]))
        self.assertEqual(self.kind_and_targets("what does shout call?"), ("callees", ["shout"]))
        self.assertEqual(self.kind_and_targets("what does store call?"), ("callees", ["Registry.store"]))  # plain word in the slot
        self.assertIsNone(self.kind_and_targets("what does main call?"))  # two `main`s: ambiguous, normal search
        self.assertEqual(self.kind_and_targets("what breaks if I change format_name?"), ("impact", ["format_name"]))
        self.assertEqual(self.kind_and_targets("does cli use pkg?"), ("between", ["cli/", "pkg/"]))
        self.assertEqual(self.kind_and_targets("who calls this?", focus=SV + "shout"), ("callers", ["shout"]))
        self.assertEqual(self.kind_and_targets("which function does shout call?"), ("callees", ["shout"]))
        self.assertEqual(self.kind_and_targets("In which components is the code in the 'pkg' folder used in?"),
                         ("callers", ["pkg/"]))
        self.assertEqual(self.kind_and_targets("where is pkg used?"), ("callers", ["pkg/"]))
        self.assertIsNone(self.kind_and_targets("how does greeting work?"))
        self.assertIsNone(self.kind_and_targets("who wrote this project?"))

    def test_answers(self):
        callers = S.answer(self.index, S.classify(self.index, "who calls format_name?"))
        sources = {f["source"] for f in callers["facts"]}
        self.assertTrue({M + "User.greet", SV + "shout", SV + "Registry.store", C + "main"} <= sources)
        self.assertIn("can miss calls", callers["summary"])  # says what static analysis can't see

        callees = S.answer(self.index, S.classify(self.index, "what does shout call?"))
        self.assertEqual({f["target"] for f in callees["facts"]}, {M + "format_name"})

        impact = S.answer(self.index, S.classify(self.index, "what depends on format_name?"))
        self.assertIn(C + "main", [f["source"] for f in impact["facts"]])

        users = S.answer(self.index, S.classify(self.index, "In which components is the code in the 'pkg' folder used in?"))
        self.assertIn("From app.py (", users["summary"])  # by component: the root file, and cli/
        self.assertIn("cli/ (", users["summary"])
        self.assertEqual({S.component_of(f["path"]) for f in users["facts"]}, {"app.py", "cli/"})

        between = S.answer(self.index, S.classify(self.index, "does cli use pkg?"))
        self.assertTrue(between["summary"].startswith("cli/ -> pkg/: 4 edge(s)"))
        self.assertIn("pkg/ -> cli/: 0 edge(s)", between["summary"])

    def test_language_server_callers_make_it_exact(self):
        self.index.lsp = lambda method, params: {"results": {n["id"]: {"supported": True, "incoming": [
            {"path": "cli/run.py", "line": 6}]} for n in params["nodes"]}}
        self.index._rebuild(lambda m: None)
        result = S.answer(self.index, S.classify(self.index, "who calls shout?"))
        self.assertTrue(result["exact"])
        self.assertIn("checked with the language server", result["summary"])

    def test_the_workflow_answers_from_the_graph(self):
        llm, steps = Fake("Yes: cli/run.py::main calls into pkg."), []
        out = Workflows(self.index, llm).ask({"query": "does cli use pkg?"}, steps.append)
        self.assertEqual([e["agent"] for e in steps if e["type"] == "agent_step"],
                         ["planner", "graph", "retriever", "analyst", "answer"])
        prompt = llm.prompts[0][1]["content"]
        self.assertTrue(prompt.startswith("Graph query (between) over the whole repository: cli/ -> pkg/: 4 edge(s)"))
        self.assertIn("### cli/run.py :: main", prompt)  # the code involved is read with it
        self.assertEqual(out["graphAnswer"]["kind"], "between")

    def test_other_questions_skip_the_graph_step(self):
        steps = []
        Workflows(self.index, Fake()).ask({"query": "how does greeting work?"}, steps.append)
        self.assertNotIn("graph", [e["agent"] for e in steps if e["type"] == "agent_step"])


class ExploreTest(Base):
    VAGUE = "which part turns things into title case?"  # names nothing; "title" isn't a symbol

    def test_gap_detection(self):
        from graphrag_code import explore as X
        state = lambda q: {"query": q, "retrieval": R.retrieve(self.index, q),
                           "context": R.retrieve(self.index, q)["context"]}
        self.assertIsNone(X.gap(self.index, state("does cli use pkg?")))  # named folders: anchored
        self.assertIsNone(X.gap(self.index, state("how does format_name work?")))  # a named symbol
        self.assertIn("best match is weak", X.gap(self.index, state("qwertyuiop asdfgh?")))

    def test_the_model_explores_with_graph_tools_then_answers(self):
        llm = Fake("format_name does it.", commands=[
            "find title case", "callers format_name", "read `pkg/models.py :: format_name`", "done"])
        steps = []
        wf = Workflows(self.index, llm)
        # Make the gap certain: a question with no anchor and a weak best match
        out = wf.ask({"query": "qwertyuiop asdfgh?"}, steps.append)
        agents = [e["agent"] for e in steps if e["type"] == "agent_step"]
        self.assertEqual(agents.count("explorer"), 4)  # three tool steps, then the step's own summary
        self.assertEqual(out["explored"], ["find title case", "callers format_name",
                                           "read pkg/models.py :: format_name", "done"])
        self.assertIn(M + "format_name", [c["id"] for c in out["context"]])
        answer_prompt = llm.prompts[0][1]["content"]
        self.assertIn("### pkg/models.py :: format_name", answer_prompt)
        self.assertIn("pkg/service.py::shout --calls--> pkg/models.py::format_name", answer_prompt)
        # Each tool result went back to the model
        conversation = "\n".join(m["content"] for m in llm.tool_prompts[-1])
        self.assertIn("Result of callers format_name:", conversation)
        self.assertIn("pkg/models.py :: User.greet (method) [calls]", conversation)

    def test_bad_replies_stop_the_exploring(self):
        llm = Fake("answer", commands=["Let me think about this.", "I will search."])
        out = Workflows(self.index, llm).ask({"query": "qwertyuiop asdfgh?"}, lambda e: None)
        self.assertEqual(out["explored"], [])
        self.assertEqual(len(llm.tool_prompts), 2)

    def test_explore_steps_zero_never_explores(self):
        llm = Fake("answer", commands=["find x"])
        out = Workflows(self.index, llm, explore_steps=0).ask({"query": "qwertyuiop asdfgh?"}, lambda e: None)
        self.assertEqual((out["explored"], llm.tool_prompts), ([], []))


if __name__ == "__main__":
    unittest.main()
