/**
 * GraphRAG Code: ask about and edit a codebase using a code graph.
 *
 * The extension host side: commands, the file watcher (incremental re-indexing), the panel,
 * the edit loop, and VS Code's language servers as a graph source (lspGraph.ts) for every
 * language the sidecar's parsers don't read. The indexing, retrieval (local push PPR) and
 * model calls run in a Python sidecar (sidecar/graphrag_code), which shares its core with the
 * GraphRAG-Search text search (core/graphrag_core).
 */
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { CheckpointStore } from './checkpoint';
import { EditFlow, PROPOSED_SCHEME, ProposalProvider } from './editFlow';
import { LspGraph, LspSymbol, NOT_CODE, languageExtensions, pool } from './lspGraph';
import { Panel } from './panel';
import { AskResult, IndexStats, Sidecar } from './sidecarClient';

const WATCH_DEBOUNCE_MS = 600;
// Folders never indexed (the sidecar skips the same ones)
const IGNORED_DIRS = ['node_modules', '.git', '.hg', '.svn', 'venv', '.venv', 'env', '__pycache__', 'build', 'dist',
  'out', 'target', 'bin', 'obj', '.gradle', '.idea', '.vscode', 'vendor', 'coverage', '.next', '.cache', '.graphrag'];
const IGNORED = new RegExp(`(^|/)(${IGNORED_DIRS.map((d) => d.replace('.', '\\.')).join('|')})(/|$)`);
const EXCLUDE_GLOB = `**/{${IGNORED_DIRS.join(',')}}/**`;
// Language-server symbols: files read at most, in parallel, per upload to the sidecar, and
// how long to wait before asking again for files whose server wasn't ready
const MAX_LSP_FILES = 2000;
const LSP_FILE_CONCURRENCY = 4;
const LSP_UPLOAD_BATCH = 25;
const LSP_RETRY_MS = 15000;

interface InitResult {
  embedder: string; parser: string; parserProblem: string; vectorStore: string; vectorStoreProblem: string;
  languageServers: boolean; parsedExtensions: string[];
}

let sidecar: Sidecar | undefined;
let starting: Promise<Sidecar> | undefined;
let parsedExtensions = new Set<string>();     // read by the sidecar's parsers
let lspLanguages = new Map<string, string>(); // extension -> language id, for everything else

function root(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function requireRoot(): string {
  const r = root();
  if (!r) throw new Error('Open a folder first: GraphRAG Code indexes the first workspace folder.');
  return r;
}

/** Where graphrag_code and graphrag_core are: the sidecar folder, then the repo's core/ when
 *  running from a checkout, else the copy `npm run bundle-core` puts next to the sidecar. */
function pythonPath(extensionPath: string): string[] {
  const sidecarDir = path.join(extensionPath, 'sidecar');
  const repoCore = path.join(extensionPath, '..', 'core');
  const bundled = path.join(sidecarDir, 'vendor');
  return [sidecarDir, fs.existsSync(path.join(repoCore, 'graphrag_core')) ? repoCore : bundled];
}

/** The interpreter: the setting, or the sidecar's own environment made by
 *  `npm run setup-python` (sidecar/.venv) when the setting is left at "python". */
function pythonExecutable(extensionPath: string, configured: string): string {
  if (configured && configured !== 'python') return configured;
  const venv = path.join(extensionPath, 'sidecar', '.venv');
  const candidates = process.platform === 'win32'
    ? [path.join(venv, 'Scripts', 'python.exe')]
    : [path.join(venv, 'bin', 'python3'), path.join(venv, 'bin', 'python')];
  return candidates.find((c) => fs.existsSync(c)) ?? 'python';
}

function summary(stats: IndexStats): string {
  const symbols = Object.values(stats.symbols).reduce((a, b) => a + b, 0);
  const edges = Object.values(stats.edges).reduce((a, b) => a + b, 0);
  const broken = stats.syntaxErrors.length ? `, ${stats.syntaxErrors.length} file(s) with syntax errors` : '';
  const langs = stats.languages?.length ? stats.languages.join('/') : 'source';
  const lsp = stats.languageServerFiles ? `, ${stats.languageServerFiles} via language servers` : '';
  return `${stats.files} ${langs} files${lsp}, ${symbols} symbols, ${edges} relationships `
    + `(${stats.parser ?? 'ast'}, ${stats.vectorStore ?? 'sqlite'}, embeddings: ${stats.embedder})${broken}`;
}

/** Workspace-relative path with '/', or undefined outside the workspace or in an ignored folder. */
function relOf(workspace: string, uri: vscode.Uri): string | undefined {
  if (uri.scheme !== 'file') return undefined;
  const rel = path.relative(workspace, uri.fsPath).split(path.sep).join('/');
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel) && !IGNORED.test(rel) ? rel : undefined;
}

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel('GraphRAG Code');
  const storage = (context.storageUri ?? context.globalStorageUri).fsPath;
  const checkpoints = new CheckpointStore(path.join(storage, 'checkpoints'));
  const proposals = new ProposalProvider();
  const panel = () => Panel.show(context.extensionUri, root);
  let fallbackWarned = false;
  let lsp: LspGraph | undefined;
  let lspRun = 0; // a newer symbol pass stops an older one

  const makeLspGraph = (workspace: string) => new LspGraph({
    exec: (command, ...args) => vscode.commands.executeCommand(command, ...args),
    uriFor: (rel) => vscode.Uri.file(path.join(workspace, ...rel.split('/'))),
    relOf: (uri) => relOf(workspace, uri),
    positionFor: (line, character) => new vscode.Position(line, character),
  });

  const lspFile = (rel: string, symbols: LspSymbol[]) =>
    ({ path: rel, lang: lspLanguages.get(path.extname(rel).toLowerCase()) ?? '', symbols });

  /**
   * In the background: symbols of every file the parsers don't read, from the language
   * servers, sent to the sidecar in batches. Files whose server returned nothing (not started
   * yet, or still loading the project) are asked once more after LSP_RETRY_MS.
   */
  const indexLanguageServerFiles = async (proc: Sidecar, workspace: string) => {
    const run = ++lspRun;
    lspLanguages = languageExtensions(vscode.extensions.all, [...parsedExtensions]);
    if (!lsp || !lspLanguages.size) return;
    const graph = lsp;
    const uris = await vscode.workspace.findFiles(`**/*{${[...lspLanguages.keys()].join(',')}}`, EXCLUDE_GLOB, MAX_LSP_FILES);
    const rels = uris.map((u) => relOf(workspace, u)).filter((r): r is string => Boolean(r));
    if (!rels.length) return;
    const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    status.show();
    let sent = 0;
    const pass = async (paths: string[]): Promise<string[]> => {
      const empty: string[] = [];
      let batch: ReturnType<typeof lspFile>[] = [];
      let done = 0;
      const upload = async () => {
        const files = batch;
        batch = [];
        if (files.length && proc.running && run === lspRun) {
          await proc.request('addSymbols', { files });
          sent += files.length;
        }
      };
      await pool(paths, LSP_FILE_CONCURRENCY, async (rel) => {
        if (run !== lspRun || !proc.running) return;
        const symbols = await graph.symbols(rel);
        done += 1;
        status.text = `$(sync~spin) GraphRAG Code: language servers ${done}/${paths.length}`;
        if (symbols.length) batch.push(lspFile(rel, symbols));
        else empty.push(rel);
        if (batch.length >= LSP_UPLOAD_BATCH) await upload();
      });
      await upload();
      return empty;
    };
    try {
      let empty = await pass(rels);
      if (empty.length && run === lspRun) {
        status.text = `$(clock) GraphRAG Code: waiting for language servers (${empty.length} files)`;
        await new Promise((r) => setTimeout(r, LSP_RETRY_MS));
        empty.forEach((rel) => graph.invalidate(rel));
        empty = await pass(empty);
      }
      output.appendLine(`Language servers: symbols from ${sent} of ${rels.length} file(s)`
        + `${empty.length ? `; nothing from ${empty.length} (no language server for them, or it isn't ready)` : ''}`);
    } catch (err: any) {
      output.appendLine(`Language-server indexing stopped: ${err.message}`);
    } finally {
      status.dispose();
    }
  };

  /** Once per session: tree-sitter or LanceDB isn't in use, and why. */
  const warnAboutFallbacks = (init: InitResult) => {
    const problems = [
      init.parser !== 'tree-sitter' ? `tree-sitter isn't available, so only Python is indexed (with ast)${init.parserProblem ? `: ${init.parserProblem}` : ''}` : '',
      init.vectorStore !== 'lancedb' && init.vectorStoreProblem ? `LanceDB isn't available, so vectors are kept in SQLite: ${init.vectorStoreProblem}` : '',
    ].filter(Boolean);
    if (!problems.length || fallbackWarned) return;
    fallbackWarned = true;
    output.appendLine(problems.join('\n'));
    void vscode.window.showWarningMessage(
      `GraphRAG Code: ${problems.map((p) => p.split(':')[0]).join('; ')}. Run "npm run setup-python" in the extension folder, `
      + 'or set graphragCode.pythonPath to a Python with sidecar/requirements.txt installed.', 'Show log')
      .then((c) => { if (c) output.show(); });
  };

  /** The running sidecar, started, initialised and indexed on first use. */
  const getSidecar = (): Promise<Sidecar> => {
    if (sidecar?.running) return Promise.resolve(sidecar);
    if (starting) return starting;
    starting = (async () => {
      const workspace = requireRoot();
      const config = vscode.workspace.getConfiguration('graphragCode');
      const python = pythonExecutable(context.extensionPath, config.get<string>('pythonPath', 'python'));
      output.appendLine(`Starting the sidecar with ${python}`);
      const proc = new Sidecar({
        python,
        pythonPath: pythonPath(context.extensionPath),
        log: (line) => output.appendLine(line),
      });
      const useLanguageServers = config.get<boolean>('languageServers', true);
      lsp = useLanguageServers ? makeLspGraph(workspace) : undefined;
      const graph = lsp;
      // The sidecar asks while its PPR, coverage check or impact analysis runs
      proc.onRequest('lsp/neighbours', (p) => (graph ? graph.neighbours(p.nodes ?? []) : { results: {} }));
      let languageServers = false;
      try {
        await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'GraphRAG Code' },
          async (progress) => {
            const report = (e: { type: string; message?: string }) => {
              if (e.type === 'status' && e.message) progress.report({ message: e.message });
            };
            const init = await proc.request<InitResult>('initialize', {
              root: workspace,
              storagePath: storage,
              embeddingModel: config.get<string>('embeddingModel', 'auto'),
              parser: config.get<string>('parser', 'auto'),
              vectorStore: config.get<string>('vectorStore', 'auto'),
              languageServers: useLanguageServers,
              exploreSteps: config.get<number>('exploreSteps', 4),
              llm: {
                baseUrl: config.get<string>('llm.baseUrl'),
                model: config.get<string>('llm.model'),
                apiKey: config.get<string>('llm.apiKey') || undefined,
              },
            }, report);
            warnAboutFallbacks(init);
            parsedExtensions = new Set(init.parsedExtensions ?? ['.py']);
            languageServers = Boolean(init.languageServers);
            const stats = await proc.request<IndexStats>('index', {}, report);
            output.appendLine(`Indexed: ${summary(stats)} in ${stats.seconds} s`);
            vscode.window.setStatusBarMessage(`GraphRAG Code: ${summary(stats)}`, 8000);
          });
      } catch (err) {
        proc.dispose();
        throw err;
      }
      sidecar = proc;
      // Other languages join in the background: questions work meanwhile
      if (languageServers) void indexLanguageServerFiles(proc, workspace);
      return proc;
    })().finally(() => { starting = undefined; });
    return starting;
  };

  // Any code file: the sidecar knows whether it has symbols for it
  const isCode = (doc: vscode.TextDocument) => doc.uri.scheme === 'file' && !NOT_CODE.has(doc.languageId);
  const focus = () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !isCode(editor.document)) return undefined;
    return { path: editor.document.uri.fsPath, line: editor.selection.active.line + 1 };
  };
  const openFiles = () => vscode.window.visibleTextEditors
    .filter((e) => isCode(e.document))
    .map((e) => e.document.uri.fsPath);

  const run = (name: string, fn: () => Promise<void>) => vscode.commands.registerCommand(name, async () => {
    try {
      await fn();
    } catch (err: any) {
      output.appendLine(`${name} failed: ${err.stack || err.message}`);
      void vscode.window.showErrorMessage(`GraphRAG Code: ${err.message.split('\n')[0]}`, 'Show log')
        .then((c) => { if (c) output.show(); });
      Panel.existing()?.note('error', err.message);
    }
  });

  const selectedText = () => {
    const editor = vscode.window.activeTextEditor;
    const text = editor && !editor.selection.isEmpty ? editor.document.getText(editor.selection) : '';
    return text.length <= 200 ? text : '';
  };

  context.subscriptions.push(
    output,
    vscode.workspace.registerTextDocumentContentProvider(PROPOSED_SCHEME, proposals),

    run('graphragCode.index', async () => {
      const proc = await getSidecar();
      const stats = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'GraphRAG Code: re-indexing' },
        (progress) => proc.request<IndexStats>('index', {}, (e) => {
          if (e.type === 'status' && e.message) progress.report({ message: e.message });
        }));
      void vscode.window.showInformationMessage(`GraphRAG Code indexed ${summary(stats)}.`);
      if (lsp) void indexLanguageServerFiles(proc, requireRoot()); // ask the language servers again too
    }),

    run('graphragCode.ask', async () => {
      const question = await vscode.window.showInputBox({
        title: 'Ask about the code',
        prompt: 'The symbol under the cursor and the symbols you name are the starting points.',
        value: selectedText(),
        ignoreFocusOut: true,
      });
      if (!question?.trim()) return;
      const at = focus();
      const proc = await getSidecar();
      panel().reset('ask', question);
      const result = await proc.request<AskResult>('ask', { query: question, focus: at, openFiles: openFiles() },
        (e) => panel().event(e));
      panel().result('ask', result);
    }),

    run('graphragCode.edit', async () => {
      const instruction = await vscode.window.showInputBox({
        title: 'Describe the change',
        prompt: 'Put the cursor in the function or class to change, or name it. You will see a diff before anything changes.',
        ignoreFocusOut: true,
      });
      if (!instruction?.trim()) return;
      await new EditFlow({ sidecar: getSidecar, root: requireRoot(), panel, proposals, checkpoints, focus, openFiles })
        .run(instruction);
    }),

    run('graphragCode.showGraph', async () => {
      const at = focus();
      if (!at) throw new Error('Put the cursor in a code file first.');
      const proc = await getSidecar();
      panel().reset('graph', `Graph around ${path.basename(at.path)}:${at.line}`);
      panel().result('graph', await proc.request('graph', { focus: at, openFiles: openFiles() }));
    }),

    run('graphragCode.impact', async () => {
      const at = focus();
      if (!at) throw new Error('Put the cursor in a function, class or component first.');
      const proc = await getSidecar();
      panel().reset('impact', `What depends on the symbol at ${path.basename(at.path)}:${at.line}`);
      panel().result('impact', await proc.request('impact', { focus: at }));
    }),

    run('graphragCode.restoreCheckpoint', async () => {
      const items = checkpoints.list().map((c) => ({
        label: c.label,
        description: new Date(c.created).toLocaleString(),
        detail: `${c.files.map((f) => f.path).join(', ')}${c.gitRef ? `  ·  ${c.gitRef}` : ''}`,
        checkpoint: c,
      }));
      if (!items.length) {
        void vscode.window.showInformationMessage('No GraphRAG Code checkpoints yet: one is taken before every applied edit.');
        return;
      }
      const pick = await vscode.window.showQuickPick(items, { title: 'Restore files to how they were before an edit' });
      if (!pick) return;
      const ok = await vscode.window.showWarningMessage(
        `Restore ${pick.checkpoint.files.length} file(s) to how they were before "${pick.label}"?`,
        { modal: true, detail: 'Changes made to these files since then are overwritten.' }, 'Restore');
      if (ok !== 'Restore') return;
      const paths = checkpoints.restore(pick.checkpoint);
      if (sidecar?.running) await sidecar.request('update', { paths });
      void vscode.window.showInformationMessage(`Restored ${paths.length} file(s).`);
    }),
  );

  // Incremental re-indexing, batched: files the parsers read are re-parsed by the sidecar;
  // other code files get their symbols from the language server again
  const changed = new Set<string>();
  const changedLsp = new Map<string, boolean>(); // path -> deleted
  let timer: NodeJS.Timeout | undefined;
  const flush = async () => {
    const proc = sidecar;
    const paths = [...changed];
    const lspPaths = [...changedLsp];
    changed.clear();
    changedLsp.clear();
    if (!proc?.running) return;
    try {
      if (paths.length) {
        const s = await proc.request<IndexStats>('update', { paths });
        output.appendLine(`Re-indexed ${paths.length} file(s): ${summary(s)}`);
      }
      if (lspPaths.length && lsp) {
        const graph = lsp;
        lspPaths.forEach(([rel]) => graph.invalidate(rel));
        const read = await pool(lspPaths, LSP_FILE_CONCURRENCY,
          async ([rel, deleted]) => (deleted ? [] : graph.symbols(rel)));
        const files = lspPaths.map(([rel], i) => lspFile(rel, read[i])).filter((f) => f.symbols.length);
        const gone = lspPaths.filter((_, i) => !read[i].length).map(([rel]) => rel);
        if (gone.length) await proc.request('removeFiles', { paths: gone });
        if (files.length) await proc.request('addSymbols', { files });
        output.appendLine(`Language servers: re-read ${lspPaths.length} file(s)`);
      }
    } catch (err: any) {
      output.appendLine(`Re-indexing failed: ${err.message}`);
    }
  };
  const queue = (uri: vscode.Uri, deleted = false) => {
    const workspace = root();
    if (!sidecar?.running || !workspace) return; // not indexed yet: the first index reads everything
    const rel = relOf(workspace, uri);
    if (!rel) return;
    const ext = path.extname(rel).toLowerCase();
    if (parsedExtensions.has(ext)) changed.add(uri.fsPath);
    else if (lsp && lspLanguages.has(ext)) changedLsp.set(rel, deleted);
    else return;
    clearTimeout(timer);
    timer = setTimeout(() => { void flush(); }, WATCH_DEBOUNCE_MS);
  };
  const watcher = vscode.workspace.createFileSystemWatcher('**/*');
  watcher.onDidChange((uri) => queue(uri));
  watcher.onDidCreate((uri) => queue(uri));
  watcher.onDidDelete((uri) => queue(uri, true));
  context.subscriptions.push(watcher);

  // New settings take effect with a fresh sidecar
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
    if (e.affectsConfiguration('graphragCode') && sidecar) {
      sidecar.dispose();
      sidecar = undefined;
    }
  }));
  context.subscriptions.push({ dispose: () => sidecar?.dispose() });
}

export function deactivate(): void {
  sidecar?.dispose();
  sidecar = undefined;
}
