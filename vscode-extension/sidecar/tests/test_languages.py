"""tree-sitter (Python parity, TypeScript/React) and LanceDB. Skipped when they aren't installed:
run with the sidecar's environment (npm run setup-python) to include them."""
import glob
import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SIDECAR, "..", ".."))
sys.path[:0] = [HERE, SIDECAR, os.path.join(REPO, "core")]

from graphrag_code import parse_treesitter as ts  # noqa: E402
from graphrag_code import parsing, retrieve as R  # noqa: E402
from graphrag_code import vectors as V  # noqa: E402
from graphrag_code.edits import apply_blocks, parse_blocks  # noqa: E402
from graphrag_code.embed import HashingEmbedder  # noqa: E402
from graphrag_code.index import Index  # noqa: E402
from graphrag_code.parse_python import parse_file as parse_with_ast  # noqa: E402
from test_sidecar import FILES, write_repo  # noqa: E402

needs_treesitter = unittest.skipUnless(ts.available(), f"tree-sitter unavailable: {ts.IMPORT_ERROR}")
needs_lancedb = unittest.skipUnless(V.lancedb is not None, f"lancedb unavailable: {V.LANCE_ERROR}")


def comparable(result):
    symbols = {s.id: (s.kind, s.start, s.end, s.doc, s.text, s.parent) for s in result.symbols}
    refs = {(r.src, r.kind, r.target, r.count) for r in result.refs}
    return symbols, refs, result.imports, bool(result.error)


@needs_treesitter
class PythonParityTest(unittest.TestCase):
    """The tree-sitter Python parser gives exactly what the ast parser gives."""

    def assert_same(self, path, source):
        a, t = parse_with_ast(path, source), ts.parse(path, source)
        sa, ra, ia, ea = comparable(a)
        st, rt, it, et = comparable(t)
        self.assertEqual(ea, et, path)
        self.assertEqual(sa, st, path)
        self.assertEqual(ra, rt, path)
        self.assertEqual(ia, it, path)

    def test_fixture(self):
        for path, text in FILES.items():
            self.assert_same(path, textwrap.dedent(text).lstrip("\n"))

    def test_this_repositorys_python(self):
        files = [f for pattern in ("query_processor/app/*.py", "core/graphrag_core/*.py",
                                   "vscode-extension/sidecar/graphrag_code/*.py", "ingestion_pipeline/*.py")
                 for f in glob.glob(os.path.join(REPO, pattern))]
        self.assertGreater(len(files), 30)
        for full in files:
            with open(full, encoding="utf-8") as fh:
                self.assert_same(os.path.relpath(full, REPO).replace(os.sep, "/"), fh.read())


TS_FILES = {
    "src/api.ts": '''
        /** Loads every user matching the query. */
        export async function fetchAll(query: string): Promise<User[]> {
          return request("/users?q=" + query);
        }

        function request(url: string) {
          return fetch(url);
        }

        export interface User { name: string }
        interface Fetcher { get(id: string): Promise<User> }
        class Base {}

        export class Client extends Base implements Fetcher {
          get(id: string) {
            return this.load(id);
          }
          load(id: string): Promise<User> {
            return fetchAll(id).then((users) => users[0]);
          }
        }
        ''',
    "src/ui/Button.tsx": '''
        import { memo } from "react";

        export interface ButtonProps { label: string }

        function Button({ label }: ButtonProps) {
          return <button>{label}</button>;
        }

        export default memo(Button);
        ''',
    "src/ui/Card.tsx": '''
        import Button from "./Button";
        import { User } from "../api";

        export const Card = ({ user }: { user: User }) => (
          <div><Button label={user.name} /></div>
        );
        ''',
    "src/ui/index.ts": '''
        export { default as Button } from "./Button";
        export * from "./Card";
        ''',
    "src/App.tsx": '''
        import React from "react";
        import * as api from "./api";
        import { Card, Button } from "./ui";

        export const Memo = React.memo(() => <Card user={{ name: "x" }} />);

        export default class App extends React.Component {
          componentDidMount() {
            api.fetchAll("x");
            new api.Client().get("1");
          }
          render() {
            return <><Card user={{ name: "a" }} /><Button label="b" /></>;
          }
        }
        ''',
    "src/broken.ts": "export function oops( {\n",
}

A, CARD, BTN, APP = "src/api.ts::", "src/ui/Card.tsx::", "src/ui/Button.tsx::", "src/App.tsx::"


@needs_treesitter
class TypeScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = os.path.join(self.tmp.name, "repo")
        write_repo(root, TS_FILES)
        self.index = Index(root, os.path.join(self.tmp.name, "s", "i.sqlite"), HashingEmbedder(), vector_store="sqlite")
        self.stats = self.index.build()
        self.g = self.index.graph

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def edge(self, src, dst, kind):
        return [e for e in self.g.out.get(src, []) if e.dst == dst and e.kind == kind]

    def test_symbols(self):
        kinds = {sid: s.kind for sid, s in self.g.symbols.items()}
        self.assertEqual(kinds[CARD + "Card"], "component")
        self.assertEqual(kinds[BTN + "Button"], "component")
        self.assertEqual(kinds[APP + "App"], "class")
        self.assertEqual(kinds[APP + "App.render"], "method")
        self.assertEqual(kinds[APP + "Memo"], "component")  # React.memo(() => <Card/>)
        self.assertEqual(kinds[A + "User"], "interface")
        self.assertEqual(self.g.symbols[A + "fetchAll"].doc, "Loads every user matching the query.")
        self.assertEqual(self.g.symbols[A + "fetchAll"].start, 2)  # "export" included
        self.assertIn("src/broken.ts", self.stats["syntaxErrors"])
        self.assertEqual(sorted(self.stats["languages"]), ["tsx", "typescript"])

    def test_relationships(self):
        self.assertTrue(self.edge(APP + "App.render", CARD + "Card", "renders"))        # via export * barrel
        self.assertTrue(self.edge(APP + "App.render", BTN + "Button", "renders"))      # default re-exported by name
        self.assertTrue(self.edge(CARD + "Card", BTN + "Button", "renders"))           # default import
        self.assertTrue(self.edge(APP + "Memo", CARD + "Card", "renders"))
        self.assertTrue(self.edge(APP + "App.componentDidMount", A + "fetchAll", "calls"))  # import * as api
        self.assertTrue(self.edge(APP + "App.componentDidMount", A + "Client", "calls"))    # new api.Client()
        self.assertTrue(self.edge(A + "Client.get", A + "Client.load", "calls"))           # this.load
        self.assertTrue(self.edge(A + "Client.load", A + "fetchAll", "calls"))
        self.assertTrue(self.edge(A + "fetchAll", A + "request", "calls"))
        self.assertTrue(self.edge(A + "Client", A + "Base", "extends"))
        self.assertTrue(self.edge(A + "Client", A + "Fetcher", "implements"))
        self.assertTrue(self.edge(A + "fetchAll", A + "User", "uses_type"))
        self.assertTrue(self.edge(CARD + "Card", A + "User", "uses_type"))
        self.assertTrue(self.edge(CARD, A + "User", "imports"))
        self.assertFalse(self.edge(A + "request", A + "fetchAll", "calls"))  # fetch() is not ours

    def test_callers_include_renderers_and_impact_follows_them(self):
        self.assertEqual(set(self.g.callers(BTN + "Button")), {APP + "App.render", CARD + "Card"})
        affected = {a["id"]: a["direct"] for a in R.impact(self.index, [BTN + "Button"])}
        self.assertTrue(affected[CARD + "Card"] and affected[APP + "App.render"])
        self.assertFalse(affected[APP + "Memo"])  # renders Card, which renders Button

    def test_edits_keep_exports_and_check_syntax(self):
        p = apply_blocks(parse_blocks(
            "=== REPLACE src/ui/Card.tsx :: Card\n"
            "const Card = ({ user }: { user: User }) => (\n  <section><Button label={user.name} /></section>\n);\n"
            "=== END"), self.index.read)
        new = p.files["src/ui/Card.tsx"]["proposed"]
        self.assertIn("export const Card = ({ user }: { user: User }) => (\n  <section>", new)
        self.assertEqual(p.changed_symbols, [CARD + "Card"])
        bad = apply_blocks(parse_blocks("=== REPLACE src/api.ts :: request\nfunction request(url {\n=== END"),
                           self.index.read)
        self.assertTrue(bad.syntax_errors and bad.syntax_errors[0].startswith("src/api.ts:"))

    def test_context_is_fenced_in_its_language(self):
        block = R.context_block(self.index, {"id": CARD + "Card"})
        self.assertIn("### src/ui/Card.tsx :: Card", block)
        self.assertIn("```tsx", block)


@needs_lancedb
class LanceDBTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = os.path.join(self.tmp.name, "repo")
        write_repo(self.root)
        self.db_path = os.path.join(self.tmp.name, "s", "i.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_search_and_incremental_embedding(self):
        index = Index(self.root, self.db_path, HashingEmbedder(), vector_store="auto")
        self.assertEqual(index.vectors.name, "lancedb")
        stats = index.build()
        self.assertEqual(stats["vectorStore"], "lancedb")
        total = sum(v for k, v in stats["symbols"].items() if k != "module")
        self.assertEqual(stats["embedded"], total)
        hits = index.vector_search("format the name title case", k=3)
        self.assertEqual(next(iter(hits)), "pkg/models.py::format_name")

        with open(os.path.join(self.root, "pkg", "service.py"), "a", encoding="utf-8") as fh:
            fh.write("\n\ndef whisper(name):\n    return shout(name).lower()\n")
        self.assertEqual(index.update(["pkg/service.py"])["embedded"], 1)
        self.assertIn("pkg/service.py::whisper", index.vector_search("whisper lower", k=2))
        index.close()

        # A restart reuses the stored vectors
        again = Index(self.root, self.db_path, HashingEmbedder(), vector_store="auto")
        self.assertEqual(again.build()["embedded"], 0)
        again.close()

    def test_another_embedding_model_gets_its_own_table(self):
        class Small(HashingEmbedder):
            name, dim = "small-8", 8

            def embed(self, texts):
                return [[1.0] + [0.0] * 7 for _ in texts]
        Index(self.root, self.db_path, HashingEmbedder()).build()
        index = Index(self.root, self.db_path, Small())
        self.assertGreater(index.build()["embedded"], 0)  # nothing reused across models
        self.assertEqual(index.vectors._table_names(), [index.vectors.table_name])  # old table dropped
        index.close()

    def test_sqlite_is_the_fallback(self):
        index = Index(self.root, self.db_path, HashingEmbedder(), vector_store="sqlite")
        self.assertEqual(index.vectors.name, "sqlite")
        index.build()
        self.assertEqual(next(iter(index.vector_search("format the name title case", k=3))),
                         "pkg/models.py::format_name")
        index.close()


class FallbackTest(unittest.TestCase):
    def test_ast_mode_reads_python_only(self):
        parsing.set_mode("ast")
        try:
            self.assertFalse(parsing.supported("src/App.tsx"))
            self.assertEqual(parsing.parse_source("m.py", "def f():\n    pass\n").parser, "ast")
        finally:
            parsing.set_mode("auto")


if __name__ == "__main__":
    unittest.main()
