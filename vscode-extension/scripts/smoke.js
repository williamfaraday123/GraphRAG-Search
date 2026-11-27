// End-to-end smoke test of the compiled stdio client (out/sidecarClient.js) against the real
// sidecar, without VS Code: start it, index a small repository, ask for the graph around a
// symbol and what depends on it, and check an unknown method is reported. No model needed.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { Sidecar } = require('../out/sidecarClient');

const repo = path.join(__dirname, '..', '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'graphrag-code-smoke-'));
const root = path.join(tmp, 'repo');
fs.mkdirSync(path.join(root, 'shop'), { recursive: true });
fs.writeFileSync(path.join(root, 'shop', '__init__.py'), '');
fs.writeFileSync(path.join(root, 'shop', 'prices.py'), [
  'def tax(amount):',
  '    return amount * 0.2',
  '',
  '',
  'def total(amount):',
  '    return amount + tax(amount)',
  '',
].join('\n'));
fs.writeFileSync(path.join(root, 'Price.tsx'), [
  'export function Price({ amount }: { amount: number }) {',
  '  return <span>{amount}</span>;',
  '}',
  '',
  'export function Basket() {',
  '  return <Price amount={1} />;',
  '}',
  '',
].join('\n'));
fs.writeFileSync(path.join(root, 'checkout.py'), [
  'from shop.prices import total',
  '',
  '',
  'class Cart:',
  '    def __init__(self):',
  '        self.items = []',
  '',
  '    def due(self):',
  '        return total(sum(self.items))',
  '',
].join('\n'));

(async () => {
  const events = [];
  const sidecar = new Sidecar({
    python: require('./python'),
    pythonPath: [path.join(repo, 'vscode-extension', 'sidecar'), path.join(repo, 'core')],
  });
  try {
    const init = await sidecar.request('initialize', { root, storagePath: path.join(tmp, 'store'), embeddingModel: 'hashing' });
    const stats = await sidecar.request('index', {}, (e) => events.push(e));
    const treeSitter = init.parser === 'tree-sitter';
    assert.strictEqual(stats.files, treeSitter ? 4 : 3);
    assert.strictEqual(stats.edges.calls, 2);
    assert.strictEqual(stats.edges.renders, treeSitter ? 1 : undefined, 'Basket renders Price');
    assert.ok(events.some((e) => e.type === 'status'), 'progress events arrive');

    const focus = { path: path.join(root, 'shop', 'prices.py'), line: 2 };
    const graph = await sidecar.request('graph', { focus });
    assert.strictEqual(graph.focus, 'shop/prices.py::tax');
    assert.ok(graph.graph.nodes.some((n) => n.id === 'shop/prices.py::total'));

    const impact = await sidecar.request('impact', { focus });
    const ids = impact.impact.map((a) => a.id);
    assert.ok(ids.includes('shop/prices.py::total'), 'direct caller');
    assert.ok(ids.includes('checkout.py::Cart.due'), 'transitive caller');

    fs.appendFileSync(path.join(root, 'checkout.py'), '\n\ndef pay(cart):\n    return cart.due()\n');
    const updated = await sidecar.request('update', { paths: [path.join(root, 'checkout.py')] });
    assert.strictEqual(updated.edges.calls, 3, 'incremental update picks up the new call');

    await assert.rejects(sidecar.request('nope'), /unknown method/);
    console.log(`smoke test passed (${init.parser}, ${init.vectorStore}):`, JSON.stringify(updated.edges));
  } finally {
    sidecar.dispose();
    setTimeout(() => fs.rmSync(tmp, { recursive: true, force: true }), 500);
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
