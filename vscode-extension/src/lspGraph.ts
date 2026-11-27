/**
 * VS Code's language servers as a source for the code graph, for any language that has a
 * language server extension installed: no per-language code.
 *
 *  - symbols(path): functions, methods, classes, interfaces... from the document symbol
 *    provider, with their exact line ranges. Sent to the sidecar for files no parser reads.
 *  - neighbours(nodes): asked by the sidecar (lsp/neighbours) when its PPR, coverage check or
 *    impact analysis reaches a node: callers and callees (call hierarchy), supertypes and
 *    subtypes (type hierarchy), and references when a server has no call hierarchy. Returned
 *    as locations; the sidecar maps each to the symbol that contains it.
 *
 * Every request has a timeout, so a slow or missing server never stalls a search; at most
 * CONCURRENCY run at once. The VS Code API comes in through `deps`, so tests can fake it.
 */

export interface LspDeps {
  /** vscode.commands.executeCommand */
  exec: (command: string, ...args: any[]) => Thenable<any>;
  /** A workspace-relative path ('src/Shop.java') -> vscode.Uri */
  uriFor: (rel: string) => any;
  /** A vscode.Uri -> workspace-relative path, or undefined outside the workspace */
  relOf: (uri: any) => string | undefined;
  /** 0-based line and character -> vscode.Position */
  positionFor: (line: number, character: number) => any;
  timeoutMs?: number;
}

export interface LspSymbol {
  name: string;
  kind: string;        // function | method | class | interface | type
  qualname: string;    // 'Cart.total' (unique within the file)
  parent: string;      // qualname of the enclosing symbol, '' at the top
  start: number;       // 1-based first and last line
  end: number;
  selLine: number;     // where the name is: 1-based line, 0-based column
  selCol: number;
}

export interface NodeRef { id: string; path: string; line: number; end?: number; name: string; kind: string; selLine?: number; selCol?: number }
export interface Location { path: string; line: number; count?: number }
export interface Relations {
  supported: boolean;  // the server has call hierarchy for this symbol (so its callers are exact)
  incoming?: Location[];
  outgoing?: Location[];
  supertypes?: Location[];
  subtypes?: Location[];
  references?: Location[];
}

// vscode.SymbolKind values -> the graph's kinds. Module/Namespace/Package (1-3) only group
// symbols; fields, variables, properties and the like aren't nodes.
export const SYMBOL_KINDS: Record<number, string> = {
  4: 'class', 5: 'method', 8: 'method', 9: 'type', 10: 'interface', 11: 'function', 22: 'class',
};
const TYPE_KINDS = new Set(['class', 'interface', 'type']);
const CONCURRENCY = 6;
const DEFAULT_TIMEOUT_MS = 4000;

function rangeOf(item: any): any {
  return item?.range ?? item?.location?.range;
}

/** Runs `fn` over `items`, at most `limit` at a time. */
export async function pool<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  });
  await Promise.all(workers);
  return out;
}

/**
 * The document symbol provider's answer -> graph nodes. Works for both shapes servers give:
 * a tree (DocumentSymbol with children) or a flat list (SymbolInformation), by nesting on
 * ranges. A name that repeats in the same scope (overloads) gets '~2', '~3'.
 */
export function flattenSymbols(raw: any[] | undefined): LspSymbol[] {
  const all: { item: any; range: any }[] = [];
  const collect = (items: any[] | undefined) => {
    for (const item of items ?? []) {
      const range = rangeOf(item);
      if (range) all.push({ item, range });
      if (item?.children?.length) collect(item.children);
    }
  };
  collect(raw);
  // Outer symbols first: by start, then the longer range
  all.sort((a, b) => a.range.start.line - b.range.start.line || b.range.end.line - a.range.end.line);
  const out: LspSymbol[] = [];
  const stack: { end: number; sym: LspSymbol }[] = [];
  const seen = new Map<string, number>();
  for (const { item, range } of all) {
    const kind = SYMBOL_KINDS[item.kind];
    const start = range.start.line + 1;
    let end = range.end.line + 1;
    if (range.end.character === 0 && end > start) end -= 1;
    while (stack.length && stack[stack.length - 1].end < start) stack.pop();
    if (!kind) continue; // not a node; its children nest under the nearest node around it
    const parent = stack.length ? stack[stack.length - 1].sym.qualname : '';
    // "(*Cart).Total" (Go receivers) -> "Total"; "save(User)" (Java parameters) -> "save"
    const name = String(item.name).replace(/^\([^)]*\)\./, '').replace(/\(.*$/, '').trim() || String(item.name);
    let qualname = parent ? `${parent}.${name}` : name;
    const n = (seen.get(qualname) ?? 0) + 1;
    seen.set(qualname, n);
    if (n > 1) qualname = `${qualname}~${n}`;
    const sel = item.selectionRange?.start ?? range.start;
    const sym: LspSymbol = { name, kind, qualname, parent, start, end, selLine: sel.line + 1, selCol: sel.character };
    out.push(sym);
    stack.push({ end, sym });
  }
  return out;
}

export class LspGraph {
  private symbolCache = new Map<string, Promise<LspSymbol[]>>();

  constructor(private readonly deps: LspDeps) {}

  private timed<T>(p: Thenable<T>): Promise<T | undefined> {
    const ms = this.deps.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(undefined), ms);
      Promise.resolve(p).then(
        (v) => { clearTimeout(timer); resolve(v); },
        () => { clearTimeout(timer); resolve(undefined); },
      );
    });
  }

  /** The file's symbols (cached until invalidate). */
  symbols(rel: string): Promise<LspSymbol[]> {
    let cached = this.symbolCache.get(rel);
    if (!cached) {
      cached = this.timed(this.deps.exec('vscode.executeDocumentSymbolProvider', this.deps.uriFor(rel)))
        .then((raw) => flattenSymbols(raw));
      this.symbolCache.set(rel, cached);
    }
    return cached;
  }

  invalidate(rel: string): void {
    this.symbolCache.delete(rel);
  }

  /** Where to ask about a node: its name, from the sidecar or found in the file's symbols. */
  private async position(node: NodeRef): Promise<any | undefined> {
    if (node.selLine) return this.deps.positionFor(node.selLine - 1, node.selCol ?? 0);
    const syms = (await this.symbols(node.path)).filter((s) => s.name === node.name);
    if (!syms.length) return undefined;
    // The parser's start line may be a decorator or `export` line just above
    const best = syms.reduce((a, b) => (Math.abs(b.start - node.line) < Math.abs(a.start - node.line) ? b : a));
    if (Math.abs(best.start - node.line) > 5 && !(best.start <= (node.end ?? node.line) && best.end >= node.line)) return undefined;
    return this.deps.positionFor(best.selLine - 1, best.selCol);
  }

  private loc(uri: any, range: any, count = 1): Location | undefined {
    const path = this.deps.relOf(uri);
    return path && range ? { path, line: range.start.line + 1, count } : undefined;
  }

  private async relations(node: NodeRef): Promise<Relations> {
    const pos = await this.position(node);
    if (!pos) return { supported: false };
    const uri = this.deps.uriFor(node.path);
    const result: Relations = { supported: false };
    const calls: any[] = (await this.timed(this.deps.exec('vscode.prepareCallHierarchy', uri, pos))) ?? [];
    if (calls.length) {
      result.supported = true;
      const item = calls[0];
      const [incoming, outgoing] = await Promise.all([
        this.timed(this.deps.exec('vscode.provideIncomingCalls', item)),
        this.timed(this.deps.exec('vscode.provideOutgoingCalls', item)),
      ]);
      result.incoming = ((incoming as any[]) ?? [])
        .map((c) => this.loc(c.from?.uri, c.from?.selectionRange ?? c.from?.range, c.fromRanges?.length || 1))
        .filter((l): l is Location => Boolean(l));
      result.outgoing = ((outgoing as any[]) ?? [])
        .map((c) => this.loc(c.to?.uri, c.to?.selectionRange ?? c.to?.range, c.fromRanges?.length || 1))
        .filter((l): l is Location => Boolean(l));
    }
    if (TYPE_KINDS.has(node.kind)) {
      const types: any[] = (await this.timed(this.deps.exec('vscode.prepareTypeHierarchy', uri, pos))) ?? [];
      if (types.length) {
        const [sup, sub] = await Promise.all([
          this.timed(this.deps.exec('vscode.provideSupertypes', types[0])),
          this.timed(this.deps.exec('vscode.provideSubtypes', types[0])),
        ]);
        const toLoc = (t: any) => this.loc(t.uri, t.selectionRange ?? t.range);
        result.supertypes = ((sup as any[]) ?? []).map(toLoc).filter((l): l is Location => Boolean(l));
        result.subtypes = ((sub as any[]) ?? []).map(toLoc).filter((l): l is Location => Boolean(l));
      }
    }
    if (!calls.length) {
      // No call hierarchy here: who refers to it (calls and type uses alike)
      const refs: any[] = (await this.timed(this.deps.exec('vscode.executeReferenceProvider', uri, pos))) ?? [];
      result.references = refs.map((r) => this.loc(r.uri, r.range)).filter((l): l is Location => Boolean(l));
    }
    return result;
  }

  /** Answers the sidecar's lsp/neighbours. */
  async neighbours(nodes: NodeRef[]): Promise<{ results: Record<string, Relations> }> {
    const answers = await pool(nodes, CONCURRENCY, (n) => this.relations(n).catch(() => ({ supported: false })));
    const results: Record<string, Relations> = {};
    nodes.forEach((n, i) => { results[n.id] = answers[i]; });
    return { results };
  }
}

// Languages that have document symbols but aren't code (headings, keys): not worth opening
export const NOT_CODE = new Set([
  'markdown', 'json', 'jsonc', 'json5', 'jsonl', 'yaml', 'xml', 'xsl', 'html', 'css', 'scss', 'less',
  'plaintext', 'ignore', 'properties', 'ini', 'toml', 'csv', 'tsv', 'log', 'diff', 'git-commit',
  'git-rebase', 'code-text-binary', 'search-result', 'snippets', 'restructuredtext', 'latex', 'bibtex',
  'tex', 'svg', 'handlebars', 'razor', 'dockerfile', 'makefile', 'pip-requirements',
]);

/** File extension -> language id, from what installed extensions contribute, leaving out
 *  the extensions a parser already reads and languages that aren't code. */
export function languageExtensions(extensions: readonly { packageJSON: any }[], parsed: string[]): Map<string, string> {
  const skip = new Set(parsed.map((e) => e.toLowerCase()));
  const map = new Map<string, string>();
  for (const ext of extensions) {
    for (const lang of ext.packageJSON?.contributes?.languages ?? []) {
      if (!lang?.id || NOT_CODE.has(lang.id)) continue;
      for (const e of lang.extensions ?? []) {
        const key = String(e).toLowerCase();
        if (key.startsWith('.') && !skip.has(key) && !map.has(key)) map.set(key, lang.id);
      }
    }
  }
  return map;
}
