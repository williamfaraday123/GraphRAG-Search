# GraphRAG Code (VS Code extension)

Ask about and edit a codebase through its code graph: Python and TypeScript/JavaScript/React parsed with tree-sitter, every other language through the language servers VS Code already runs. This is GraphRAG-Search's retrieval applied to code: symbols instead of text chunks, real code relationships instead of model-extracted facts, and an edit loop after the answer.

It's a separate package in the same repository. It shares a small core with the text search ([`core/graphrag_core`](../core/graphrag_core)): local push PPR, hybrid ranking, the coverage check, the workflow skeleton, the progress events and the chat client. The text search's pipeline has no code mode.

## How it works

| Text search | This extension |
|---|---|
| A chunk of a document | A symbol: module, class, function, method, React component, interface or type alias |
| RELATION / FACT edges extracted by a model (about 4 s per sentence, with checks for made-up facts) | Static analysis with tree-sitter: `calls`, `renders` (component to child), `imports`, `extends`, `implements`, `contains`, `uses_type`. Deterministic: this repository's React client (17 files) indexes in 0.2 s |
| Neo4j, Milvus, MinIO | LanceDB (an embedded vector database) for the symbol embeddings, SQLite for the parse results, both in the extension's storage; the graph is kept in memory in a local sidecar process |
| MiniLM text embeddings | A code-search embedding model when `sentence-transformers` is installed, else identifier hashing (no download) |
| Edge weights from confidence | Direction-aware weights (caller to callee 1.0, back 0.7; subclass to base 0.8, back 0.5; ...), scaled by how often a symbol calls another |
| Seeds: entities the question names, vector hits | The same, plus the symbol under the cursor and the open files, so context follows what you're working on |
| Coverage: every entity the question names is in context | Also: before changing a function, every caller of it is in context |

**Ask** (planner → [graph] → retriever → analyst → [explorer] → answer): local push PPR from the seeds over the code graph, ranked with similarity (45%), PPR (40%) and being named in the question (15%). The model answers from that context, citing `path:line`, which you can click.

- **Starting points:** a name counts at full weight only when it's written as code (`push_ppr`, `LocalPPR`, `local_ppr.py`, or anything in backticks). An ordinary word that happens to be a name, like "running" matching `Sidecar.running`, is a weak hint. Folder names ("core", "vscode-extension") start from the modules in those folders, and each named folder gets some of its code into the context. With two named folders, the code connecting them comes first.
- **Relationships:** beside the code, the model reads edges from the graph. That's every edge between two named folders, plus the edges across files among the best-ranked symbols (for example `retrieve.py::retrieve --calls--> core/graphrag_core/ppr.py::push_ppr`). This is the code version of the knowledge-graph facts the text search gives its model, and the panel lists them under "Relationships the model read". The model is told never to conclude that something doesn't happen just because the code shown doesn't show it.
- **Structural questions are answered by the graph** (`structural.py`): "who calls X / where is X used", "what does X call", "what depends on X / what breaks if I change X", and "does A use B". X can be a symbol, a folder, or the code under the cursor ("who calls this?"). The graph's answer covers the whole repository, not just the dozen symbols a context holds. It's exact for what static analysis resolved, says what it can miss (calls through objects of unknown type), and is checked by the language server when one is available. The model explains the result with the code of the symbols involved, and the panel shows it under "From the code graph".
- **Exploring, only when there's a gap:** when the question names nothing to start from and even the best match is weak, or a symbol it's about didn't fit in the context, the model gets up to `graphragCode.exploreSteps` (default 4) turns with graph tools before answering: `find`, `callers`, `callees`, `read`. Each step is one line, one model call and one graph hop, and what it reads joins the context. A small model follows the format, but it doesn't always explore well; set `exploreSteps` to 0 to answer every question from one search.
- **Folder nodes:** every folder with indexed files is a node, with light `contains_dir` edges to its subfolders and modules. Code in the same folder counts as a little related, and the graph view shows the project's layout. Clicking a folder shows it in the Explorer.

**Edit** (planner → retriever → analyst → editor → impact, then verify in VS Code):

1. The context includes every caller of the symbols being changed (coverage check).
2. The model rewrites whole symbols in `=== REPLACE path :: Qualified.name` blocks, which small models get right far more often than diffs. The sidecar places them at the symbols' current lines with the right indentation. A proposal that doesn't parse goes back to the model once with the syntax error.
3. **Impact analysis**: PPR from the changed symbols, pushed only backwards along dependencies (callers, subclasses, type users, importers), lists what is likely to break.
4. Each file opens as a diff, and nothing changes until you approve.
5. A **checkpoint** is taken. The files' exact contents are saved, and in a git repository it's also a commit under `refs/graphrag/checkpoints/<id>`, built with a temporary index so your branch, index and working tree are untouched. Then the change is applied as one `WorkspaceEdit` and saved.
6. **Verify**: new errors from the language server in the changed files and in the dependents, then `graphragCode.verify.testCommand` if set. Errors that existed before the change, or tests that already failed, don't count. Problems go back to the model for another proposal, up to `verify.maxAttempts`, and each proposal is previewed and approved again.

**Show Graph Around Cursor** and **Show What Depends on This Symbol** need no model: they draw the PPR neighbourhood as a graph (node colour and size show the PPR score; components are hexagons, interfaces and types diamonds). Clicking a node opens the code.

## Languages

| | Python | TypeScript / TSX / JavaScript / JSX |
|---|---|---|
| Symbols | modules, classes, functions, methods | functions (including `const f = () => ...` and `React.memo`/`forwardRef` wrappers), classes, methods and arrow-function fields, interfaces, type aliases; a capitalised function that returns JSX is a component |
| Calls | `f()`, `self.m()`, `super().m()`, `module.f()`, constructors | `f()`, `this.m()`, `api.f()`, `new C()` |
| Imports | absolute and relative, through re-exporting `__init__.py` | relative imports, `export * from` barrels, `export { default as X } from`, default exports (also `export default memo(X)`) |
| Types | annotations, including string annotations | type annotations, `extends`, `implements` |

Both languages are parsed by tree-sitter. If tree-sitter isn't installed or can't load, Python falls back to the standard library's `ast`, which gives exactly the same result (a test compares them over every Python file in this repository). Likewise, without LanceDB the vectors are kept in SQLite and searched in memory. The extension says once which fallback it is using and why.

### Every other language: VS Code's language servers

Java, Go, C#, Rust, C/C++, Kotlin, PHP, Ruby and the rest work through the language servers VS Code already runs, with no per-language code. Any language with a language server extension installed is covered.

- **Symbols:** in the background after the first index, the extension reads every other code file's symbols with `vscode.executeDocumentSymbolProvider`. These are functions, methods, classes, interfaces and types with their exact line ranges, and the sidecar embeds them and makes them searchable like parsed ones. Files whose server isn't ready yet are asked again once, 15 seconds later. **Re-index Workspace** asks again, and the file watcher re-reads changed files.
- **Edges, on demand:** push PPR only asks for the neighbours of the nodes it reaches, so the whole graph never has to be built. When a search, the coverage check or impact analysis reaches one of these symbols, the sidecar asks the extension (`lsp/neighbours`, a request in the other direction over stdio). The extension answers with:
  - callers and callees from call hierarchy;
  - supertypes and subtypes from type hierarchy;
  - references, as a `uses` edge, when a server has no call hierarchy.

  Answers are cached per symbol until a file they involve changes. A first question in a new part of the code costs a few seconds; repeat questions are instant. Every request has a timeout (4 s), so a slow server never stalls a search.
- **Exact callers before an edit:** in any language, including Python and TypeScript, the language server's callers of the symbols being changed settle the parser's guesses. A guessed edge it confirms stays (no longer marked as a guess), one it contradicts is dropped, and callers the parser missed are added. Edges the parser resolved for certain are kept. The analyst step reports what changed.
- **Edits** work the same way: symbols are replaced by their line ranges, and the language server's diagnostics check the result.

What a language server gives depends on the server. Strongly typed languages (Java, C#, Go, Rust, TypeScript) give exact edges; some servers need setup first (clangd needs `compile_commands.json`, Java needs the project imported). There are no `imports` or `renders` edges for these files: language servers don't report imports, and component rendering is framework knowledge. Turn it off with `graphragCode.languageServers`.

## Setup

Requirements:
- **Python 3.10+.**
- **An OpenAI-compatible chat endpoint.** By default Ollama at `http://localhost:11434/v1` with `qwen3:4b-instruct`.

From this folder:
```
npm install
npm run compile
npm run setup-python
```
`setup-python` creates the sidecar's own environment, `sidecar/.venv`, with [sidecar/requirements.txt](sidecar/requirements.txt) (tree-sitter and its grammars, LanceDB, numpy). It then checks that they load: Windows Smart App Control can block a package's compiled modules. The extension uses this environment while `graphragCode.pythonPath` is left at `python`. To use a code-aware embedding model, also install `sentence-transformers` there.

Then open `vscode-extension/` in VS Code and press F5 (Run Extension). In the new window, open a project and run **GraphRAG Code: Ask About the Code** (also in the editor's right-click menu). The first command indexes the workspace. After that, a file watcher re-indexes changed files incrementally.

To build a `.vsix`, run `npm run package`. It copies the shared core into `sidecar/vendor` first (`npm run bundle-core`). The `.vsix` doesn't contain the Python environment, so run `setup-python` (or point `pythonPath` at a Python with the requirements) wherever it's installed.

Settings (`graphragCode.*`): `pythonPath`, `parser` (`auto` or `ast`), `languageServers` (on by default), `exploreSteps` (4; 0 turns exploring off), `vectorStore` (`auto`, `lancedb` or `sqlite`), `llm.baseUrl`, `llm.model`, `llm.apiKey`, `embeddingModel`, `verify.testCommand`, `verify.maxAttempts`, `verify.diagnosticsTimeoutSeconds`.

## Tests

```
npm test
```
This runs:
- the shared core's tests;
- the sidecar's tests: parsing (including tree-sitter/ast parity and a TypeScript/React project), resolution, incremental updates, LanceDB, retrieval, coverage, impact, the edit format in both languages, both workflows with a fake model, and the stdio protocol;
- questions about structure: names written as code versus plain words, folders as starting points, folder nodes, the relationships in the prompt, structural questions (classification, answers, the language-server check, the workflow), and exploring with graph tools (gap detection, a scripted model, malformed replies, turning it off);
- language servers with a fake standing in for VS Code (sidecar side): symbols from a Kotlin file, edges loaded lazily and cached until the file changes, guessed callers confirmed, dropped and added, an edit, and the sidecar asking the extension over stdio;
- a Node smoke test of the compiled client against the real sidecar;
- a Node test of `lspGraph.ts` against a fake VS Code: both shapes of document symbols, call and type hierarchy, the references fallback, timeouts, and the full path for a Go file (the real sidecar asks the compiled client while it computes impact);
- a test of git checkpoints against a throwaway repository.

No model is needed. The tests use `sidecar/.venv` when it exists. `PYTHON=python npm test` runs them with plain Python instead: the tree-sitter and LanceDB tests are skipped and the fallbacks are tested. The parts that need a running VS Code (the panel, diff previews, `WorkspaceEdit`, the diagnostics check) have no automated tests yet.

## Limits

- **Parsed languages vs. language servers.** Python and TypeScript/JavaScript are parsed: the whole graph is ready in milliseconds, with `imports` and `renders` edges. Other languages depend on their language server: no edges until a search reaches their symbols (a few seconds the first time), and only what the server reports. A parser module for a language (see `sidecar/graphrag_code/parse_treesitter.py`) makes it faster and richer, but isn't needed for it to work. Language-server symbols are read for at most 2,000 files per workspace. In JS/TS, non-relative imports (packages, and `tsconfig` path aliases such as `@/components`) and `require()` aren't followed by the parser.
- **Dynamic calls are incomplete.** A call on a receiver of unknown type (`obj.save()`) is linked to a method of that name only when there are at most 3 in the repository, as a "guessed" edge at half weight. Guessed edges don't count as callers for the coverage check. Calls into packages outside the repository are not followed.
- **Large repositories.** Re-indexing a file re-parses only that file, but the graph is re-resolved from every file's stored parse result. That takes milliseconds for hundreds of files, but very large monorepos would need incremental resolution.
- **Edit quality depends on the model.** Small local models handle a few symbols. With `qwen3:4b-instruct`, a Python signature change with its caller worked, and so did a TSX change: a new prop on a component and its interface, passed from the component that renders it. The model also tends to rewrite symbols it wasn't asked to touch (unchanged rewrites are ignored, real ones show up in the diff). Use a larger model for bigger multi-file changes.
- **Not yet evaluated.** Aider's repo map already ranks context with a graph. This extension's differences are question-centred PPR (from the cursor and the question, not a global ranking), the coverage check and the visual graph. Whether those give better answers and edits needs an evaluation on real tasks.

To try it

In vscode-extension/, run npm install, npm run compile and npm run setup-python.
Open that folder in VS Code and press F5.
In the new window, open a project in any language and run "GraphRAG Code: Ask About the Code". Python and TypeScript/JavaScript are parsed; other languages need their language server extension installed (for example "Extension Pack for Java" or "Go").