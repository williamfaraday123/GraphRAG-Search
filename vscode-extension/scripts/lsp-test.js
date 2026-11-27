// Tests out/lspGraph.js against a fake VS Code (document symbols, call and type hierarchy,
// references), then the whole path without VS Code: the real sidecar asks the compiled client
// for lsp/neighbours while it computes impact, and the client answers from the fake.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { LspGraph, flattenSymbols, languageExtensions } = require('../out/lspGraph');
const { Sidecar } = require('../out/sidecarClient');

const R = (l1, c1, l2, c2) => ({ start: { line: l1, character: c1 }, end: { line: l2, character: c2 } });

// --- flattenSymbols ------------------------------------------------------------------------
// A tree (DocumentSymbol), as most servers give: a namespace (not a node), a class with a
// field (not a node), two overloads, and a Go-style receiver name
const tree = [{
  name: 'shop', kind: 2, range: R(0, 0, 30, 0), selectionRange: R(0, 10, 0, 14), children: [
    { name: 'Cart', kind: 4, range: R(2, 0, 12, 1), selectionRange: R(2, 13, 2, 17), children: [
      { name: 'items', kind: 7, range: R(3, 2, 3, 20), selectionRange: R(3, 10, 3, 15), children: [] },
      { name: 'add(Item)', kind: 5, range: R(5, 2, 7, 3), selectionRange: R(5, 7, 5, 10), children: [] },
      { name: 'add(Item, int)', kind: 5, range: R(9, 2, 11, 3), selectionRange: R(9, 7, 9, 10), children: [] },
    ] },
    { name: '(*Cart).Total', kind: 5, range: R(14, 0, 16, 1), selectionRange: R(14, 15, 14, 20), children: [] },
  ],
}];
const flat = flattenSymbols(tree);
assert.deepStrictEqual(flat.map((s) => s.qualname), ['Cart', 'Cart.add', 'Cart.add~2', 'Total']);
assert.deepStrictEqual(flat[1], { name: 'add', kind: 'method', qualname: 'Cart.add', parent: 'Cart', start: 6, end: 8, selLine: 6, selCol: 7 });
assert.strictEqual(flat[3].parent, '');

// A flat list (SymbolInformation): nesting comes from the ranges
const list = flattenSymbols([
  { name: 'Shop', kind: 4, location: { range: R(0, 0, 9, 1) } },
  { name: 'open', kind: 5, location: { range: R(2, 2, 4, 3) } },
  { name: 'helper', kind: 11, location: { range: R(11, 0, 13, 1) } },
]);
assert.deepStrictEqual(list.map((s) => [s.qualname, s.parent, s.kind]),
  [['Shop', '', 'class'], ['Shop.open', 'Shop', 'method'], ['helper', '', 'function']]);

// --- languageExtensions ----------------------------------------------------------------------
const langs = languageExtensions([
  { packageJSON: { contributes: { languages: [{ id: 'go', extensions: ['.go'] }, { id: 'markdown', extensions: ['.md'] }] } } },
  { packageJSON: { contributes: { languages: [{ id: 'python', extensions: ['.py', '.pyi'] }] } } },
  { packageJSON: {} },
], ['.py']);
assert.deepStrictEqual([...langs], [['.go', 'go'], ['.pyi', 'python']]);

// --- a fake VS Code ---------------------------------------------------------------------------
const GO = [
  'package shop',              // 1
  '',
  'type Cart struct {',        // 3
  '\tItems []int',
  '}',
  '',
  'func (c *Cart) Total() int {', // 7
  '\treturn Sum(c.Items) + Tax(Sum(c.Items))',
  '}',
  '',
  'func Tax(amount int) int {',  // 11
  '\treturn amount / 5',
  '}',
  '',
  'func Sum(xs []int) int {',    // 15
  '\tt := 0',
  '\tfor _, x := range xs { t += x }',
  '\treturn t',
  '}',
  '',
].join('\n');
const goSymbols = [
  { name: 'Cart', kind: 22, range: R(2, 0, 4, 1), selectionRange: R(2, 5, 2, 9), children: [] },
  { name: '(*Cart).Total', kind: 5, range: R(6, 0, 8, 1), selectionRange: R(6, 15, 6, 20), children: [] },
  { name: 'Tax', kind: 11, range: R(10, 0, 12, 1), selectionRange: R(10, 5, 10, 8), children: [] },
  { name: 'Sum', kind: 11, range: R(14, 0, 18, 1), selectionRange: R(14, 5, 14, 8), children: [] },
];
const F = 'shop/cart.go';
const item = (line) => {
  const s = goSymbols.find((g) => g.range.start.line <= line && line <= g.range.end.line);
  return s && { name: s.name, uri: { rel: F }, range: s.range, selectionRange: s.selectionRange };
};
const asked = [];
function fakeExec(neverAnswer = new Set()) {
  return (command, ...args) => {
    asked.push(command);
    if (neverAnswer.has(command)) return new Promise(() => {});
    switch (command) {
      case 'vscode.executeDocumentSymbolProvider':
        return Promise.resolve(args[0].rel === F ? goSymbols : []);
      case 'vscode.prepareCallHierarchy': {
        const it = item(args[1].line);
        return Promise.resolve(it && it.name !== 'Cart' ? [it] : []);
      }
      case 'vscode.provideIncomingCalls': {
        const callers = { Tax: [7], Sum: [7, 7], '(*Cart).Total': [] }[args[0].name] || [];
        return Promise.resolve(callers.map((l) => ({ from: item(l - 1), fromRanges: [R(l - 1, 0, l - 1, 1)] })));
      }
      case 'vscode.provideOutgoingCalls': {
        const callees = { '(*Cart).Total': [11, 15] }[args[0].name] || [];
        return Promise.resolve(callees.map((l) => ({ to: item(l - 1), fromRanges: [R(7, 0, 7, 1)] })));
      }
      case 'vscode.prepareTypeHierarchy':
        return Promise.resolve([]);
      case 'vscode.executeReferenceProvider': // Cart: used in Total's receiver
        return Promise.resolve([{ uri: { rel: F }, range: R(6, 9, 6, 13) }, { uri: { rel: 'outside' }, range: R(0, 0, 0, 1) }]);
      default:
        return Promise.reject(new Error(`unexpected ${command}`));
    }
  };
}
const deps = (exec, timeoutMs) => ({
  exec, timeoutMs, uriFor: (rel) => ({ rel }), relOf: (u) => (u.rel === 'outside' ? undefined : u.rel),
  positionFor: (line, character) => ({ line, character }),
});

(async () => {
  // --- neighbours ------------------------------------------------------------------------------
  const lsp = new LspGraph(deps(fakeExec()));
  const syms = await lsp.symbols(F);
  assert.deepStrictEqual(syms.map((s) => s.qualname), ['Cart', 'Total', 'Tax', 'Sum']);
  const { results } = await lsp.neighbours([
    { id: `${F}::Total`, path: F, line: 7, name: 'Total', kind: 'method', selLine: 7, selCol: 15 },
    { id: `${F}::Tax`, path: F, line: 11, name: 'Tax', kind: 'function' }, // no position: found by name
    { id: `${F}::Cart`, path: F, line: 3, name: 'Cart', kind: 'class' },
    { id: 'gone.go::x', path: 'gone.go', line: 1, name: 'x', kind: 'function' },
  ]);
  assert.deepStrictEqual(results[`${F}::Total`].outgoing, [{ path: F, line: 11, count: 1 }, { path: F, line: 15, count: 1 }]);
  assert.deepStrictEqual(results[`${F}::Tax`], { supported: true, incoming: [{ path: F, line: 7, count: 1 }], outgoing: [] });
  assert.strictEqual(results[`${F}::Cart`].supported, false); // no call hierarchy for a type...
  assert.deepStrictEqual(results[`${F}::Cart`].references, [{ path: F, line: 7, count: 1 }]); // ...references instead
  assert.deepStrictEqual(results['gone.go::x'], { supported: false });
  assert.strictEqual(asked.filter((c) => c === 'vscode.executeDocumentSymbolProvider').length, 2); // cached per file

  // A server that never answers costs the timeout, not a hang
  const slow = new LspGraph(deps(fakeExec(new Set(['vscode.prepareCallHierarchy', 'vscode.executeReferenceProvider'])), 50));
  const started = Date.now();
  const r = await slow.neighbours([{ id: 'a', path: F, line: 11, name: 'Tax', kind: 'function', selLine: 11, selCol: 5 }]);
  assert.deepStrictEqual(r.results.a, { supported: false, references: [] });
  assert.ok(Date.now() - started < 1000);

  // --- through the real sidecar ------------------------------------------------------------------
  const repo = path.join(__dirname, '..', '..');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'graphrag-lsp-'));
  const root = path.join(tmp, 'repo');
  fs.mkdirSync(path.join(root, 'shop'), { recursive: true });
  fs.writeFileSync(path.join(root, ...F.split('/')), GO);
  const sidecar = new Sidecar({
    python: require('./python'),
    pythonPath: [path.join(repo, 'vscode-extension', 'sidecar'), path.join(repo, 'core')],
  });
  const graph = new LspGraph(deps(fakeExec()));
  let neighbourRequests = 0;
  sidecar.onRequest('lsp/neighbours', (p) => { neighbourRequests += 1; return graph.neighbours(p.nodes); });
  try {
    const init = await sidecar.request('initialize', { root, storagePath: path.join(tmp, 's'), embeddingModel: 'hashing',
      vectorStore: 'sqlite', languageServers: true });
    assert.ok(init.languageServers && !init.parsedExtensions.includes('.go'));
    await sidecar.request('index', {});
    const added = await sidecar.request('addSymbols', { files: [{ path: F, lang: 'go', symbols: await graph.symbols(F) }] });
    assert.strictEqual(added.languageServerFiles, 1);
    assert.strictEqual(neighbourRequests, 0, 'nothing asked before a search reaches the file');

    const impact = await sidecar.request('impact', { symbols: [`${F}::Sum`] });
    const byId = Object.fromEntries(impact.impact.map((a) => [a.id, a.direct]));
    assert.strictEqual(byId[`${F}::Total`], true, 'Total calls Sum (call hierarchy, through the client)');
    assert.ok(neighbourRequests > 0);

    const view = await sidecar.request('graph', { focus: { path: path.join(root, 'shop', 'cart.go'), line: 8 } });
    assert.strictEqual(view.focus, `${F}::Total`);
    const kinds = new Set(view.graph.edges.map((e) => `${e.source.split('::')[1]}-${e.kind}-${e.target.split('::')[1]}`));
    assert.ok(kinds.has('Total-calls-Tax') && kinds.has('Total-calls-Sum'), [...kinds].join(' '));
    console.log(`language-server test passed (${neighbourRequests} lsp/neighbours requests)`);
  } finally {
    sidecar.dispose();
    setTimeout(() => fs.rmSync(tmp, { recursive: true, force: true }), 500);
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
