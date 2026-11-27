"""Language servers as a graph source, with a fake standing in for VS Code: symbols of files no
parser reads, edges loaded lazily as PPR reaches them (and cached), guessed callers settled
before an edit, and the sidecar asking the extension over stdio."""
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
sys.path[:0] = [HERE, SIDECAR, CORE]

from graphrag_code import retrieve as R  # noqa: E402
from graphrag_code.embed import HashingEmbedder  # noqa: E402
from graphrag_code.index import Index  # noqa: E402
from graphrag_code.workflows import Workflows  # noqa: E402
from test_sidecar import FakeLLM, write_repo  # noqa: E402

KOTLIN = '''
    class Cart(val items: List<Int>) {
        fun total(): Int {
            return items.sum() + tax(items.sum())
        }
    }

    fun tax(amount: Int): Int {
        return amount / 5
    }

    fun checkout(cart: Cart): Int = cart.total()
    '''
K = "shop/Cart.kt"
# What VS Code's symbol provider would report (1-based lines; selLine/selCol: where the name is)
KOTLIN_SYMBOLS = [
    {"name": "Cart", "kind": "class", "qualname": "Cart", "parent": "", "start": 1, "end": 5, "selLine": 1, "selCol": 6},
    {"name": "total", "kind": "method", "qualname": "Cart.total", "parent": "Cart", "start": 2, "end": 4, "selLine": 2, "selCol": 8},
    {"name": "tax", "kind": "function", "qualname": "tax", "parent": "", "start": 7, "end": 9, "selLine": 7, "selCol": 4},
    {"name": "checkout", "kind": "function", "qualname": "checkout", "parent": "", "start": 11, "end": 11, "selLine": 11, "selCol": 4},
]
# What call hierarchy would answer: locations of the related symbols
KOTLIN_RELATIONS = {
    f"{K}::Cart.total": {"supported": True, "incoming": [{"path": K, "line": 11}], "outgoing": [{"path": K, "line": 7}]},
    f"{K}::tax": {"supported": True, "incoming": [{"path": K, "line": 2, "count": 2}], "outgoing": []},
    f"{K}::checkout": {"supported": True, "incoming": [], "outgoing": [{"path": K, "line": 2}]},
    f"{K}::Cart": {"supported": False, "references": [{"path": K, "line": 11}]},
}


class FakeVSCode:
    """Answers lsp/neighbours like the extension would, and records what it was asked."""

    def __init__(self, relations):
        self.relations, self.asked = relations, []

    def __call__(self, method, params):
        assert method == "lsp/neighbours"
        ids = [n["id"] for n in params["nodes"]]
        self.asked += ids
        return {"results": {i: self.relations.get(i, {"supported": True}) for i in ids}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = os.path.join(self.tmp.name, "repo")
        write_repo(self.root)
        write_repo(self.root, {K: KOTLIN})
        self.db = os.path.join(self.tmp.name, "s", "i.sqlite")
        self.vscode = FakeVSCode(KOTLIN_RELATIONS)
        self.index = self.open()

    def open(self):
        index = Index(self.root, self.db, HashingEmbedder(), vector_store="sqlite")
        index.lsp = self.vscode
        index.build()
        return index

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def add_kotlin(self):
        return self.index.add_external([{"path": K, "lang": "kotlin", "symbols": KOTLIN_SYMBOLS}])


class LazyEdgesTest(Base):
    def test_symbols_become_searchable_nodes(self):
        stats = self.add_kotlin()
        self.assertEqual(stats["languageServerFiles"], 1)
        g = self.index.graph
        self.assertEqual(g.symbols[f"{K}::Cart.total"].kind, "method")
        self.assertIn("fun total(): Int {", g.symbols[f"{K}::Cart.total"].text)
        self.assertTrue([e for e in g.out[f"{K}::Cart"] if e.dst == f"{K}::Cart.total" and e.kind == "contains"])
        self.assertIn(f"{K}::tax", self.index.vector_search("tax amount", k=3))
        self.assertEqual(self.vscode.asked, [])  # nothing asked until a search reaches them
        self.assertIn("```kotlin", R.context_block(self.index, {"id": f"{K}::tax"}))

    def test_ppr_loads_edges_as_it_reaches_nodes_and_caches_them(self):
        self.add_kotlin()
        r = R.retrieve(self.index, "", focus={"path": K, "line": 11})  # in checkout
        self.assertEqual(r["focus"], f"{K}::checkout")
        self.assertIn(f"{K}::tax", r["ppr"])  # checkout -> total -> tax, two hops away
        g = self.index.graph
        self.assertTrue([e for e in g.out[f"{K}::Cart.total"] if e.dst == f"{K}::tax" and e.kind == "calls"])
        self.assertEqual(g.callers(f"{K}::tax"), [f"{K}::Cart.total"])
        self.assertIn(f"{K}::checkout", g.callers(f"{K}::Cart"))  # references: a "uses" edge
        asked = list(self.vscode.asked)
        self.assertNotIn("pkg/models.py::User", asked)  # parsed files never ask the language server

        R.retrieve(self.index, "", focus={"path": K, "line": 11})
        self.assertEqual(self.vscode.asked, asked)  # second search: all from the cache

        self.index.update([K])  # the file changed: its cached edges are stale
        R.retrieve(self.index, "", focus={"path": K, "line": 11})
        self.assertGreater(len(self.vscode.asked), len(asked))

    def test_impact_and_coverage_use_the_lazy_edges(self):
        self.add_kotlin()
        affected = {a["id"]: a["direct"] for a in R.impact(self.index, [f"{K}::tax"])}
        self.assertTrue(affected[f"{K}::Cart.total"])
        self.assertFalse(affected[f"{K}::checkout"])  # calls total, which calls tax
        r = R.retrieve(self.index, "change tax", focus={"path": K, "line": 8}, mode="edit", k=1)
        self.assertIn(f"{K}::Cart.total", [c["id"] for c in r["context"]])

    def test_symbols_survive_a_restart_until_the_file_changes(self):
        self.add_kotlin()
        self.index.close()
        self.index = self.open()
        self.assertIn(f"{K}::tax", self.index.graph.symbols)
        with open(os.path.join(self.root, *K.split("/")), "a", encoding="utf-8") as fh:
            fh.write("\nfun more() = 1\n")
        self.index.close()
        self.index = self.open()  # changed since: dropped until VS Code sends its symbols again
        self.assertNotIn(f"{K}::tax", self.index.graph.symbols)

    def test_removed_files_and_unanswered_requests(self):
        self.add_kotlin()
        self.index.lsp = lambda method, params: None  # VS Code didn't answer in time
        self.assertEqual(self.index.graph.callers(f"{K}::tax"), [])
        self.index.lsp = self.vscode  # asked again next time: nothing was cached
        self.index.remove_files([K])
        self.assertNotIn(f"{K}::tax", self.index.graph.symbols)


S, M, A = "pkg/service.py::", "pkg/models.py::", "app.py::"


class SettleCallersTest(Base):
    def test_exact_callers_confirm_drop_and_add(self):
        g = self.index.graph
        guessed = [e for e in g.into[M + "Base.save"] if e.src == S + "create_user"]
        self.assertTrue(guessed and guessed[0].guessed)
        self.vscode.relations = {
            M + "Base.save": {"supported": True, "incoming": []},                        # the guess was wrong
            S + "Registry.register": {"supported": True, "incoming": [{"path": "app.py", "line": 6}]},  # right
            M + "format_name": {"supported": True, "incoming": [
                {"path": "pkg/models.py", "line": 15}, {"path": "pkg/service.py", "line": 11},
                {"path": "pkg/service.py", "line": 20}, {"path": "app.py", "line": 6}]},  # main: the parser missed it
        }
        result = self.index.settle_callers([M + "Base.save", S + "Registry.register", M + "format_name"])
        self.assertEqual(result, {"checked": 3, "confirmed": 1, "dropped": 1, "added": 1})
        self.assertNotIn(S + "create_user", [e.src for e in g.into[M + "Base.save"]])
        self.assertFalse([e for e in g.into[S + "Registry.register"] if e.src == A + "main"][0].guessed)
        self.assertIn(A + "main", g.callers(M + "format_name"))

    def test_without_call_hierarchy_the_parsers_view_stays(self):
        self.vscode.relations = {M + "Base.save": {"supported": False}}
        self.assertEqual(self.index.settle_callers([M + "Base.save"])["checked"], 0)
        self.assertTrue([e for e in self.index.graph.into[M + "Base.save"] if e.src == S + "create_user"])

    def test_edit_retrieval_reports_it(self):
        self.vscode.relations = {M + "Base.save": {"supported": True, "incoming": []}}
        r = R.retrieve(self.index, "change Base.save", mode="edit")
        self.assertEqual(r["settled"]["dropped"], 1)


class EditTest(Base):
    def test_editing_a_file_only_a_language_server_reads(self):
        self.add_kotlin()
        llm = FakeLLM(["=== REPLACE shop/Cart.kt :: tax\nfun tax(amount: Int): Int {\n    return amount / 4\n}\n=== END"])
        result = Workflows(self.index, llm).edit({"instruction": "make tax 25%",
                                                  "focus": {"path": K, "line": 8}}, lambda e: None)
        self.assertEqual([f["path"] for f in result["files"]], [K])
        self.assertIn("return amount / 4", result["files"][0]["proposed"])
        self.assertEqual(result["changedSymbols"], [f"{K}::tax"])
        self.assertIn(f"{K}::Cart.total", [a["id"] for a in result["impact"] if a["direct"]])
        self.assertIn("### shop/Cart.kt :: Cart.total", llm.prompts[0][1]["content"])  # its caller, in context


class ProtocolTest(unittest.TestCase):
    def test_the_sidecar_asks_the_extension_mid_request(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = os.path.join(tmp, "repo")
            write_repo(root, {K: KOTLIN})
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([SIDECAR, CORE])}
            proc = subprocess.Popen([sys.executable, "-m", "graphrag_code"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                    encoding="utf-8", env=env)

            def send(obj):
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()

            def result_of(rid):
                """Reads until the response to `rid`, answering the sidecar's own requests."""
                while True:
                    msg = json.loads(proc.stdout.readline())
                    if msg.get("method") == "lsp/neighbours":
                        ids = [n["id"] for n in msg["params"]["nodes"]]
                        send({"id": msg["id"], "result": {"results": {i: KOTLIN_RELATIONS.get(i, {}) for i in ids}}})
                    elif msg.get("id") == rid:
                        return msg
            try:
                send({"id": 1, "method": "initialize", "params": {"root": root, "storagePath": os.path.join(tmp, "s"),
                                                                  "embeddingModel": "hashing", "vectorStore": "sqlite",
                                                                  "languageServers": True}})
                init = result_of(1)["result"]
                self.assertTrue(init["languageServers"])
                self.assertIn(".py", init["parsedExtensions"])
                send({"id": 2, "method": "index", "params": {}})
                result_of(2)
                send({"id": 3, "method": "addSymbols", "params": {"files": [{"path": K, "lang": "kotlin", "symbols": KOTLIN_SYMBOLS}]}})
                self.assertEqual(result_of(3)["result"]["languageServerFiles"], 1)
                send({"id": 4, "method": "impact", "params": {"symbols": [f"{K}::tax"]}})
                impact = result_of(4)["result"]["impact"]
                self.assertIn(f"{K}::Cart.total", [a["id"] for a in impact])
                send({"method": "shutdown"})
            finally:
                proc.stdin.close()
                proc.wait(timeout=10)
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
