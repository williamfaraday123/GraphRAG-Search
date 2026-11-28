"""The architecture graph (architecture.py): components from build files and compose services,
Leiden clusters when there are none, code and runtime connections (HTTP, gRPC, processes,
datastores, compose), the drill-down views, top-down retrieval and impact across components."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.dirname(HERE)
sys.path[:0] = [HERE, SIDECAR, os.path.abspath(os.path.join(SIDECAR, "..", "..", "core"))]

from graphrag_code import architecture as A  # noqa: E402
from graphrag_code import miniyaml  # noqa: E402
from graphrag_code import retrieve as R  # noqa: E402
from graphrag_code import structural as S  # noqa: E402
from graphrag_code.embed import HashingEmbedder  # noqa: E402
from graphrag_code.index import Index  # noqa: E402
from graphrag_code.workflows import Workflows  # noqa: E402
from test_sidecar import write_repo  # noqa: E402
from test_structure import Fake  # noqa: E402

SYSTEM = {
    "web/package.json": '{"name": "web", "description": "Browser client for items"}',
    "web/src/api.js": '''
        const BASE = process.env.API;

        export async function createItem(item) {
          return fetch(`${BASE}/api/items`, { method: 'POST', body: JSON.stringify(item) });
        }

        export async function deleteItem(id) {
          return fetch(`${BASE}/api/items/${id}`, { method: 'DELETE' });
        }
        ''',
    "server/requirements.txt": "flask\nredis\n",
    "server/app.py": '''
        import redis
        from flask import Flask
        from shared.lib import normalise
        import search_pb2_grpc

        app = Flask(__name__)
        cache = redis.Redis(host="cache")


        @app.route("/api/items", methods=["POST"])
        def create_item():
            return normalise("x")


        @app.delete("/api/items/<item_id>")
        def delete_item(item_id):
            stub = search_pb2_grpc.SearchServiceStub(None)
            return stub.Find(item_id)
        ''',
    "search/pyproject.toml": '[project]\nname = "search"\ndescription = "Full-text search over items"\n',
    "search/service.py": '''
        import search_pb2_grpc


        class Searcher(search_pb2_grpc.SearchServiceServicer):
            def Find(self, request, context):
                return request


        def serve(server):
            search_pb2_grpc.add_SearchServiceServicer_to_server(Searcher(), server)
        ''',
    "search/search.proto": '''
        syntax = "proto3";
        service SearchService {
          rpc Find (Query) returns (Result);
        }
        ''',
    "shared/pyproject.toml": '[project]\nname = "shared"\n',
    "shared/lib.py": '''
        def normalise(text):
            return text.strip().lower()
        ''',
    "cli/package.json": '{"name": "cli"}',
    "cli/index.js": '''
        const { spawn } = require('child_process');
        // spawn('python', ['-m', 'shared.lib']) in a comment is not a connection
        spawn('python', ['-u', '-m', 'server.app']);
        ''',
    "docker-compose.yml": '''
        services:
          web:
            build: ./web
            depends_on: [api]
          api:
            build:
              context: ./server
            environment:
              - REDIS_URL=redis://cache:6379
            depends_on:
              cache:
                condition: service_started
          cache:
            image: redis:7
        ''',
}

# No build files: two tightly knit groups of modules and one call between them
CLUSTERS = {}
for group in ("alpha", "beta"):
    for i in range(3):
        others = [f"{group}{j}" for j in range(3) if j != i]
        CLUSTERS[f"src/{group}{i}.py"] = (
            "".join(f"from src.{o} import f_{o}\n" for o in others)
            + f"\n\ndef f_{group}{i}():\n" + "".join(f"    f_{o}()\n" for o in others) + "    return 1\n")
CLUSTERS["src/alpha0.py"] += "\n\nfrom src.beta0 import f_beta0\n\n\ndef bridge():\n    return f_beta0()\n"


class Architecture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = os.path.join(self.tmp.name, "repo")
        write_repo(self.root, SYSTEM)
        self.index = Index(self.root, os.path.join(self.tmp.name, "s", "i.sqlite"), HashingEmbedder(), vector_store="sqlite")
        self.index.build()
        self.arch = A.build(self.index, "auto")

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def conn(self, src, dst, kind):
        return next((c for c in self.arch.connections if (c["from"], c["to"], c["kind"]) == (src, dst, kind)), None)

    def test_components_from_build_files_and_compose(self):
        self.assertEqual(self.arch.method, "manifests")
        code = {c["id"] for c in self.arch.code_components()}
        self.assertEqual(code, {"web", "server", "search", "shared", "cli"})
        self.assertEqual(self.arch.components["web"]["description"], "Browser client for items")
        self.assertEqual(self.arch.components["search"]["description"], "Full-text search over items")
        self.assertIn("api", self.arch.components["server"]["services"])
        redis = self.arch.components["infra:redis"]
        self.assertEqual(redis["kind"], "datastore")
        self.assertIn("cache", redis["name"])  # the compose service's name

    def test_http_clients_matched_to_routes(self):
        http = self.conn("web", "server", "http")
        self.assertIsNotNone(http)
        self.assertEqual(set(http["endpoints"]), {"POST /api/items", "DELETE /api/items/*"})
        self.assertTrue(any(e["path"] == "web/src/api.js" for e in http["evidence"]))
        self.assertTrue(any(e["path"] == "server/app.py" for e in http["evidence"]))

    def test_grpc_process_datastore_code_and_compose(self):
        grpc = self.conn("server", "search", "grpc")
        self.assertIsNotNone(grpc)
        self.assertIn("Find", grpc["label"])               # the rpcs from the .proto
        process = self.conn("cli", "server", "process")
        self.assertIsNotNone(process)
        self.assertEqual(len([c for c in self.arch.connections if c["kind"] == "process"]), 1)  # not the comment
        self.assertIsNotNone(self.conn("server", "infra:redis", "datastore"))
        code = self.conn("server", "shared", "code")
        self.assertIsNotNone(code)
        self.assertGreaterEqual(code["kinds"].get("calls", 0), 1)
        compose = self.conn("server", "infra:redis", "compose")
        self.assertIn("REDIS_URL", compose["label"])
        self.assertIsNotNone(self.conn("web", "server", "compose"))  # depends_on: [api]

    def test_views_and_mermaid(self):
        system = A.view(self.index, self.arch)
        self.assertEqual(system["level"], "system")
        self.assertTrue({"web", "server", "infra:redis"} <= {n["id"] for n in system["nodes"]})
        self.assertIn("flowchart LR", system["mermaid"])
        inner = A.view(self.index, self.arch, "server")
        ids = {n["id"] for n in inner["nodes"]}
        self.assertIn("file:server/app.py", ids)
        self.assertIn("shared", ids)  # a neighbouring component, at the end of its edge
        self.assertTrue(any(e["source"] == "file:server/app.py" and e["target"] == "shared" for e in inner["edges"]))

    def test_cross_component_impact(self):
        g = self.index.graph
        changed = ["server/app.py::delete_item"]
        across = A.cross_impact(self.index, self.arch, changed)
        self.assertEqual(len(across), 1)
        self.assertEqual(across[0]["path"], "web/src/api.js")
        self.assertIn("DELETE /api/items/*", across[0]["via"])
        servicer = A.cross_impact(self.index, self.arch, ["search/service.py::Searcher.Find"])
        self.assertEqual([a["path"] for a in servicer], ["server/app.py"])
        self.assertIn("server/app.py::delete_item", g.symbols)

    def test_top_down_ranking_and_boundaries(self):
        scores = A.rank_components(self.index, self.arch, "how does the web client talk to the server?", {})
        top = [c for c in scores if self.arch.components[c]["kind"] == "code"][:2]
        self.assertEqual(set(top), {"web", "server"})
        boundary = A.boundary_symbols(self.index, self.arch, ["web", "server"])
        self.assertIn("server/app.py::delete_item", boundary)
        ret = R.retrieve(self.index, "how does the web client talk to the server?", arch=self.arch)
        self.assertIn("web", [c["id"] for c in ret["components"]])
        self.assertIn("web --http: POST /api/items", ret["architecture"])
        self.assertIn("server/app.py::delete_item", [c["id"] for c in ret["context"]])

    def test_architecture_question_answered_from_component_graph(self):
        llm = Fake()
        steps = []
        out = Workflows(self.index, llm, architecture=lambda: self.arch).ask(
            {"query": "Give me an overview of the system architecture"}, steps.append)
        self.assertEqual(out["graphAnswer"]["kind"], "architecture")
        prompt = llm.prompts[0][1]["content"]
        self.assertIn("web --http: POST /api/items", prompt)
        self.assertIn("server --grpc: gRPC SearchService (Find)", prompt)

    def test_folder_question_gets_runtime_connections(self):
        q = S.classify(self.index, "which code uses the server folder?")
        result = S.answer(self.index, q, self.arch)
        kinds = {f["kind"] for f in result["facts"]}
        self.assertTrue({"http", "process"} <= kinds, result["summary"])

    def test_cached_until_the_graph_changes(self):
        first = A.get(self.index)
        self.assertIs(A.get(self.index), first)
        self.index.update([os.path.join(self.root, "shared", "lib.py")])
        self.assertIsNot(A.get(self.index), first)

    def test_summaries_cached_by_input(self):
        storage = os.path.join(self.tmp.name, "s")

        class Model:
            calls = 0

            def invoke(self, messages):
                Model.calls += 1
                return type("R", (), {"content": "<think>x</think>Does things."})()
        A.summarize(self.index, self.arch, Model(), storage, lambda m: None)
        first = Model.calls
        self.assertEqual(first, len(self.arch.code_components()))
        self.assertEqual(self.arch.components["web"]["summary"], "Does things.")
        A.summarize(self.index, self.arch, Model(), storage, lambda m: None)
        self.assertEqual(Model.calls, first)  # nothing changed: no model calls
        self.assertEqual(A.load_summaries(storage)["web"], "Does things.")


class Clusters(unittest.TestCase):
    def test_leiden_finds_the_two_groups(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = os.path.join(tmp, "repo")
            write_repo(root, CLUSTERS)
            index = Index(root, os.path.join(tmp, "s", "i.sqlite"), HashingEmbedder(), vector_store="sqlite")
            index.build()
            arch = A.build(index, "auto")
            self.assertEqual(arch.method, "clusters")
            groups = {arch.component_of(f"src/{g}{i}.py") for g in ("alpha", "beta") for i in range(3)}
            self.assertEqual(len(groups), 2)
            self.assertEqual(len({arch.component_of(f"src/alpha{i}.py") for i in range(3)}), 1)
            self.assertIsNotNone(next(c for c in arch.connections if c["kind"] == "code"))
            index.close()


class MiniYaml(unittest.TestCase):
    def test_compose_subset(self):
        import textwrap
        text = textwrap.dedent('''
            services:
              db:
                image: "postgres:16"   # comment
                ports: ["5432:5432"]
                environment:
                  - A=b:c
              app:
                build:
                  context: ./app
                depends_on:
                  db:
                    condition: service_healthy
                command: >
                  run
                  fast
            ''')
        import builtins
        real = builtins.__import__

        def no_yaml(name, *a, **k):  # the fallback parser, even where PyYAML is installed
            if name == "yaml":
                raise ImportError
            return real(name, *a, **k)
        builtins.__import__ = no_yaml
        try:
            data = miniyaml.load(text)
        finally:
            builtins.__import__ = real
        db, app = data["services"]["db"], data["services"]["app"]
        self.assertEqual(db["image"], "postgres:16")
        self.assertEqual(db["ports"], ["5432:5432"])
        self.assertEqual(db["environment"], ["A=b:c"])
        self.assertEqual(app["build"]["context"], "./app")
        self.assertEqual(app["depends_on"]["db"]["condition"], "service_healthy")
        self.assertEqual(app["command"], "run fast")


if __name__ == "__main__":
    unittest.main()
