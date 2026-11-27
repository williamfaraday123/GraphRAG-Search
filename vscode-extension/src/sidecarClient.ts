/**
 * Talks to the Python sidecar (sidecar/graphrag_code/server.py) over stdio: one JSON message
 * per line. Requests get a response; while one runs, the sidecar streams progress events
 * (status, agent_step, chunk: the same types as the web client's) for it.
 *
 * No `vscode` import here, so plain Node can test it (scripts/smoke.js).
 */
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import * as path from 'path';
import * as readline from 'readline';

export interface SidecarEvent {
  type: 'status' | 'agent_step' | 'chunk' | 'done' | 'error';
  stage?: string;
  message?: string;
  agent?: string;
  detail?: string;
  content?: string;
}
export type EventHandler = (event: SidecarEvent) => void;

export interface SidecarOptions {
  python: string;          // interpreter, e.g. "python" or a venv's python
  pythonPath: string[];    // directories holding graphrag_code and graphrag_core
  log?: (line: string) => void;
}

interface Pending {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
  onEvent?: EventHandler;
}

const STDERR_LINES_KEPT = 30;

export type RequestHandler = (params: any) => unknown | Promise<unknown>;

export class Sidecar {
  private proc: ChildProcessWithoutNullStreams;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private handlers = new Map<string, RequestHandler>();
  private stderrTail: string[] = [];
  private exited = false;

  constructor(private readonly opts: SidecarOptions) {
    const env = {
      ...process.env,
      PYTHONPATH: [...opts.pythonPath, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      PYTHONIOENCODING: 'utf-8',
      PYTHONUNBUFFERED: '1',
    };
    this.proc = spawn(opts.python, ['-u', '-m', 'graphrag_code'], { env });
    readline.createInterface({ input: this.proc.stdout }).on('line', (line) => this.onLine(line));
    readline.createInterface({ input: this.proc.stderr }).on('line', (line) => {
      this.stderrTail = [...this.stderrTail, line].slice(-STDERR_LINES_KEPT);
      opts.log?.(line);
    });
    this.proc.on('error', (err) => this.fail(`Could not start ${opts.python}: ${err.message}`));
    this.proc.on('exit', (code) => this.fail(`The sidecar exited (code ${code}).`));
  }

  get running(): boolean {
    return !this.exited;
  }

  request<T = any>(method: string, params: object = {}, onEvent?: EventHandler): Promise<T> {
    if (this.exited) return Promise.reject(new Error('The sidecar is not running.'));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.proc.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
    });
  }

  /** Answers the sidecar's own requests (e.g. lsp/neighbours) while it handles one of ours. */
  onRequest(method: string, handler: RequestHandler): void {
    this.handlers.set(method, handler);
  }

  private write(obj: object): void {
    if (!this.exited) this.proc.stdin.write(`${JSON.stringify(obj)}\n`);
  }

  private answer(msg: any): void {
    const handler = this.handlers.get(msg.method);
    if (!handler) {
      this.write({ id: msg.id, error: { message: `unknown method ${msg.method}` } });
      return;
    }
    Promise.resolve()
      .then(() => handler(msg.params ?? {}))
      .then((result) => this.write({ id: msg.id, result }))
      .catch((err) => this.write({ id: msg.id, error: { message: String(err?.message ?? err) } }));
  }

  dispose(): void {
    if (this.exited) return;
    try {
      this.proc.stdin.write(`${JSON.stringify({ method: 'shutdown' })}\n`);
      this.proc.stdin.end();
    } catch {
      /* already closed */
    }
    const proc = this.proc;
    setTimeout(() => { if (!this.exited) proc.kill(); }, 2000).unref?.();
  }

  private onLine(line: string): void {
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      this.opts.log?.(`(not JSON) ${line}`);
      return;
    }
    if (msg.method === 'event') {
      this.pending.get(msg.params?.id)?.onEvent?.(msg.params.event);
      return;
    }
    if (msg.method && msg.id !== undefined) { // the sidecar asking us
      this.answer(msg);
      return;
    }
    const pending = this.pending.get(msg.id);
    if (!pending) return;
    this.pending.delete(msg.id);
    if (msg.error) pending.reject(new Error(msg.error.message || 'The sidecar reported an error.'));
    else pending.resolve(msg.result);
  }

  private fail(reason: string): void {
    if (this.exited) return;
    this.exited = true;
    const tail = this.stderrTail.slice(-8).join('\n');
    const err = new Error(tail ? `${reason}\n${tail}` : reason);
    this.pending.forEach((p) => p.reject(err));
    this.pending.clear();
  }
}

// --- Result shapes (sidecar/graphrag_code) -----------------------------------------------

export interface GraphNode { id: string; label: string; kind: string; path: string; line: number; score: number; highlight: boolean }
export interface GraphEdge { source: string; target: string; kind: string; count: number; guessed: boolean }
export interface GraphData { nodes: GraphNode[]; edges: GraphEdge[] }
export interface ContextItem {
  id: string; path: string; start: number; end: number; kind: string; qualname: string;
  score: number; semantic: number; ppr: number; named: number; reason: string;
}
export interface ImpactItem { id: string; path: string; line: number; qualname: string; kind: string; score: number; direct: boolean }
export interface Relation { source: string; target: string; kind: string; guessed: boolean; text: string; path?: string; line?: number }
export interface GraphAnswer { kind: string; summary: string; facts: Relation[]; exact: boolean }
export interface AskResult {
  answer: string; context: ContextItem[]; graph: GraphData;
  relations?: Relation[]; folderLinks?: { from: string; to: string; count: number; text: string }[];
  graphAnswer?: GraphAnswer | null;
}
export interface ProposedFile { path: string; original: string; proposed: string; created: boolean }
export interface EditResult {
  summary: string;
  files: ProposedFile[];
  changedSymbols: string[];
  skipped: string[];
  syntaxErrors: string[];
  impact: ImpactItem[];
  context: ContextItem[];
  graph: GraphData;
}
export interface IndexStats {
  files: number; symbols: Record<string, number>; edges: Record<string, number>;
  unresolved: number; syntaxErrors: string[]; embedded: number; embedder: string;
  parsed?: number; seconds?: number;
  languages?: string[];        // languages found in the workspace
  parser?: string;             // "tree-sitter" or "ast"
  vectorStore?: string;        // "lancedb" or "sqlite"
  languageServerFiles?: number; // files whose symbols came from VS Code's language servers
}
