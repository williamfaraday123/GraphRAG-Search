"""Run from vscode-extension/sidecar with ../../core on PYTHONPATH:
    python -m unittest discover -s tests
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.dirname(HERE)
CORE = os.path.abspath(os.path.join(SIDECAR, "..", "..", "core"))
sys.path[:0] = [SIDECAR, CORE]

from graphrag_code import retrieve as R  # noqa: E402
from graphrag_code.edits import apply_blocks, parse_blocks  # noqa: E402
from graphrag_code.embed import HashingEmbedder, identifier_words  # noqa: E402
from graphrag_code.index import Index  # noqa: E402
from graphrag_code.parse_python import parse_file  # noqa: E402
from graphrag_code.workflows import Workflows  # noqa: E402

FILES = {
    "pkg/__init__.py": "from .models import User\n",
    "pkg/models.py": '''
        class Base:
            def save(self):
                return self.validate()

            def validate(self):
                return True


        class User(Base):
            """A person who signs in."""
            def __init__(self, name: str):
                self.name = name

            def greet(self) -> str:
                return format_name(self.name)


        def format_name(name):
            return name.title()
        ''',
    "pkg/service.py": '''
        from pkg.models import User, format_name
        import pkg.models as m


        def create_user(name) -> User:
            user = User(name)
            user.save()
            return user


        def shout(name):
            return format_name(name).upper()


        class Registry:
            def register(self, name):
                u = create_user(name)
                self.store(u)

            def store(self, user: "User"):
                m.format_name(user.name)
        ''',
    "app.py": '''
        from pkg import User
        from pkg.service import Registry


        def main():
            Registry().register("ada")
            return User("x")
        ''',
    "broken.py": "def oops(:\n",
}


def write_repo(root, files=FILES):
    for rel, text in files.items():
        full = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(text).lstrip("\n"))


class Repo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "repo")
        write_repo(self.root)
        self.index = Index(self.root, os.path.join(self.tmp.name, "store", "index.sqlite"), HashingEmbedder())
        self.stats = self.index.build()
        self.g = self.index.graph

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def edge(self, src, dst, kind):
        return [e for e in self.index.graph.out.get(src, []) if e.dst == dst and e.kind == kind]


S, M, A = "pkg/service.py::", "pkg/models.py::", "app.py::"


class GraphTest(Repo):
    def test_symbols_and_syntax_errors(self):
        self.assertIn(M + "User.greet", self.g.symbols)
        self.assertEqual(self.g.symbols[M + "User.greet"].kind, "method")
        self.assertEqual(self.g.symbols[M + "User"].doc, "A person who signs in.")
        self.assertEqual(self.stats["syntaxErrors"], ["broken.py"])

    def test_cursor_outside_named_symbol_falls_back_to_module(self):
        result = R.retrieve(
            self.index,
            "",
            {"path": os.path.join(self.root, "pkg", "models.py"), "line": 999},
        )
        self.assertEqual(result["focus"], M)
        view = R.graph_view(self.index, result["ppr"], [result["focus"]])
        self.assertIn(M, {node["id"] for node in view["nodes"]})

    def test_resolved_relationships(self):
        self.assertTrue(self.edge(S + "create_user", M + "User", "calls"))          # constructor
        self.assertTrue(self.edge(M + "User", M + "Base", "extends"))
        self.assertTrue(self.edge(M + "Base.save", M + "Base.validate", "calls"))   # self.method
        self.assertTrue(self.edge(M + "User.greet", M + "format_name", "calls"))    # same module
        self.assertTrue(self.edge(S + "shout", M + "format_name", "calls"))         # from-import
        self.assertTrue(self.edge(S + "Registry.store", M + "format_name", "calls"))  # import as m
        self.assertTrue(self.edge(S + "Registry.store", M + "User", "uses_type"))   # "User" string
        self.assertTrue(self.edge(S + "create_user", M + "User", "uses_type"))      # return type
        self.assertTrue(self.edge(S + "Registry.register", S + "Registry.store", "calls"))
        self.assertTrue(self.edge(M + "User", M + "User.greet", "contains"))
        self.assertTrue(self.edge(A, M + "User", "imports"))                         # via pkg/__init__
        self.assertTrue(self.edge(A + "main", M + "User", "calls"))

    def test_unknown_receivers_are_guessed_at_half_weight(self):
        guessed = self.edge(S + "create_user", M + "Base.save", "calls")
        self.assertTrue(guessed and guessed[0].guessed)
        self.assertTrue(self.edge(A + "main", S + "Registry.register", "calls")[0].guessed)
        self.assertNotIn(S + "create_user", self.g.callers(M + "Base.save"))  # guesses aren't callers

    def test_callers_and_seed_candidates(self):
        self.assertEqual(set(self.g.callers(M + "format_name")),
                         {M + "User.greet", S + "shout", S + "Registry.store"})
        seeds = self.g.seed_candidates("Why does Registry.register call create_user?")
        self.assertEqual(set(seeds), {S + "Registry.register", S + "create_user"})
        self.assertEqual(self.g.symbol_at("pkg/models.py", 15), M + "User.greet")

    def test_incremental_update(self):
        with open(os.path.join(self.root, "pkg", "service.py"), "a", encoding="utf-8") as fh:
            fh.write("\n\ndef loud(name):\n    return shout(name)\n")
        os.remove(os.path.join(self.root, "broken.py"))
        stats = self.index.update(["pkg/service.py", os.path.join(self.root, "broken.py")])
        self.assertTrue(self.edge(S + "loud", S + "shout", "calls"))
        self.assertEqual(stats["syntaxErrors"], [])
        self.assertEqual(stats["embedded"], 1)  # only the new symbol


class RetrieveTest(Repo):
    def test_cursor_and_named_symbols_lead_the_context(self):
        r = R.retrieve(self.index, "what does format_name do", focus={"path": "pkg/service.py", "line": 11})
        ids = [c["id"] for c in r["context"]]
        self.assertEqual(r["focus"], S + "shout")
        self.assertEqual(set(ids[:2]), {S + "shout", M + "format_name"})
        self.assertGreater(r["ppr"][M + "format_name"], r["ppr"].get(M + "Base.validate", 0))

    def test_edit_mode_adds_every_caller(self):
        r = R.retrieve(self.index, "rename format_name to normalise_name", mode="edit", k=1)
        ids = {c["id"] for c in r["context"]}
        self.assertTrue({M + "User.greet", S + "shout", S + "Registry.store"} <= ids)
        reasons = {c["id"]: c["reason"] for c in r["context"]}
        self.assertIn("coverage check", reasons[S + "shout"])

    def test_impact_lists_direct_dependents_first(self):
        affected = R.impact(self.index, [M + "format_name"])
        direct = [a["id"] for a in affected if a["direct"]]
        # Its callers, and the module that imports it (a rename breaks the import)
        self.assertEqual(set(direct), {M + "User.greet", S + "shout", S + "Registry.store", S})
        self.assertTrue(all(a["direct"] for a in affected[:len(direct)]))
        self.assertIn(S + "Registry.register", [a["id"] for a in affected])  # calls store: transitive

    def test_graph_view(self):
        view = R.graph_view(self.index, {M + "format_name": 1.0, S + "shout": 0.5}, [S + "shout"])
        self.assertEqual({n["id"] for n in view["nodes"]}, {M + "format_name", S + "shout"})
        self.assertEqual(view["edges"][0]["kind"], "calls")


class EditsTest(unittest.TestCase):
    SRC = textwrap.dedent('''
        class A:
            @property
            def x(self):
                return 1

            def y(self):
                return 2


        def f():
            return A().x
        ''').lstrip()

    def apply(self, reply, files=None):
        files = files or {"m.py": self.SRC}
        return apply_blocks(parse_blocks(reply), files.get)

    def test_replace_keeps_indentation_and_tracks_changed_symbols(self):
        p = self.apply("Made x return 10.\n=== REPLACE m.py :: A.x\n```python\n@property\ndef x(self):\n"
                       "    return 10\n```\n=== END\n")
        new = p.files["m.py"]["proposed"]
        self.assertIn("    @property\n    def x(self):\n        return 10\n", new)
        self.assertIn("    def y(self):\n        return 2\n", new)
        self.assertEqual(p.changed_symbols, ["m.py::A.x"])
        self.assertEqual(p.syntax_errors, [])

    def test_insert_append_create_short_names_and_errors(self):
        p = self.apply("=== INSERT_AFTER m.py :: y\ndef z(self):\n    return 3\n=== END\n"
                       "=== APPEND m.py\nVALUE = f()\n=== END\n"
                       "=== CREATE new.py\nimport m\n=== END\n"
                       "=== REPLACE m.py :: nope\nx\n=== END\n")
        new = p.files["m.py"]["proposed"]
        self.assertIn("        return 2\n\n    def z(self):\n        return 3\n", new)
        self.assertTrue(new.endswith("\nVALUE = f()\n"))
        self.assertTrue(p.files["new.py"]["created"])
        self.assertIn("m.py::A.z", p.changed_symbols)
        self.assertEqual(p.skipped, ["REPLACE m.py :: nope: no such symbol"])

    def test_replacing_a_symbol_with_the_same_code_changes_nothing(self):
        p = self.apply("=== REPLACE m.py :: A.y\ndef y(self):\n    return 2\n=== END\n"
                       "=== REPLACE m.py :: f\ndef f():\n    return A().y\n=== END")
        self.assertEqual(p.changed_symbols, ["m.py::f"])

    def test_a_line_range_instead_of_a_name_finds_the_symbol(self):
        # What a small model wrote when the context headers were "path:start-end"
        p = self.apply("=== REPLACE m.py:10-11\ndef f():\n    return A().y()\n=== END")
        self.assertIn("return A().y()", p.files["m.py"]["proposed"])
        self.assertEqual(p.changed_symbols, ["m.py::f"])

    def test_syntax_errors_are_reported(self):
        p = self.apply("=== REPLACE m.py :: f\ndef f(:\n    pass\n=== END")
        self.assertTrue(p.syntax_errors and p.syntax_errors[0].startswith("m.py:"))


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def stream(self, messages):
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        for i in range(0, len(reply), 7):
            yield reply[i:i + 7]


class WorkflowsTest(Repo):
    def test_ask_streams_the_answer_with_events(self):
        llm, seen = FakeLLM(["`format_name` title-cases (`pkg/models.py:18`)."]), []
        result = Workflows(self.index, llm).ask({"query": "what does format_name do?"}, seen.append)
        self.assertIn("title-cases", result["answer"])
        self.assertEqual([e["agent"] for e in seen if e["type"] == "agent_step"],
                         ["planner", "retriever", "analyst", "answer"])
        self.assertIn("def format_name(name):", llm.prompts[0][1]["content"])
        self.assertTrue(result["graph"]["nodes"])

    def test_edit_retries_on_syntax_errors_then_reports_impact(self):
        bad = "=== REPLACE pkg/models.py :: format_name\ndef format_name(name:\n=== END"
        good = ("Strip spaces too.\n=== REPLACE pkg/models.py :: format_name\n"
                "def format_name(name):\n    return name.strip().title()\n=== END")
        llm, seen = FakeLLM([bad, good]), []
        result = Workflows(self.index, llm).edit({"instruction": "make format_name strip spaces"}, seen.append)
        self.assertEqual(result["summary"], "Strip spaces too.")
        self.assertEqual([f["path"] for f in result["files"]], ["pkg/models.py"])
        self.assertIn("name.strip().title()", result["files"][0]["proposed"])
        self.assertIn("That didn't work", llm.prompts[1][-1]["content"])
        self.assertIn(S + "shout", [a["id"] for a in result["impact"] if a["direct"]])
        # The callers were in the context the model saw (coverage check)
        self.assertIn("def shout(name):", llm.prompts[0][1]["content"])


class ProtocolTest(unittest.TestCase):
    def test_stdio_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "repo")
            write_repo(root)
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([SIDECAR, CORE])}
            requests = [
                {"id": 1, "method": "initialize", "params": {"root": root, "storagePath": os.path.join(tmp, "s"),
                                                             "embeddingModel": "hashing"}},
                {"id": 2, "method": "index", "params": {}},
                {"id": 3, "method": "impact", "params": {"focus": {"path": os.path.join(root, "pkg", "models.py"), "line": 19}}},
                {"id": 4, "method": "nope"},
                {"method": "shutdown"},
            ]
            out = subprocess.run([sys.executable, "-m", "graphrag_code"], input="\n".join(map(json.dumps, requests)) + "\n",
                                 capture_output=True, text=True, env=env, timeout=60, encoding="utf-8")
            messages = [json.loads(l) for l in out.stdout.splitlines()]
            results = {m["id"]: m for m in messages if "id" in m}
            self.assertEqual(results[2]["result"]["symbols"]["class"], 3)
            self.assertIn(S + "shout", [a["id"] for a in results[3]["result"]["impact"]])
            self.assertIn("unknown method", results[4]["error"]["message"])
            self.assertTrue(any(m.get("method") == "event" and m["params"]["id"] == 2 for m in messages))


class EmbedTest(unittest.TestCase):
    def test_identifier_words(self):
        self.assertEqual(identifier_words("parseHTTPConfig_file self"), ["parse", "http", "config", "file"])


if __name__ == "__main__":
    unittest.main()
