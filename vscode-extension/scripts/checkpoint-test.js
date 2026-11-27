// Checks checkpoints (out/checkpoint.js) against a throwaway git repository: the git ref holds
// the files as they were (including uncommitted changes), the user's branch, index and working
// tree are untouched, and restore puts edited files back and deletes created ones.
const assert = require('assert');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { CheckpointStore } = require('../out/checkpoint');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'graphrag-checkpoint-'));
const root = path.join(tmp, 'repo');
fs.mkdirSync(root);
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
git('init', '-q');
git('config', 'user.email', 'test@example.invalid');
git('config', 'user.name', 'test');
fs.writeFileSync(path.join(root, 'a.py'), 'x = 1\n');
git('add', 'a.py');
git('commit', '-q', '-m', 'first');
fs.writeFileSync(path.join(root, 'a.py'), 'x = 2  # uncommitted\n');
fs.writeFileSync(path.join(root, 'staged.py'), 's = 1\n');
git('add', 'staged.py');
const statusBefore = git('status', '--porcelain');
const headBefore = git('rev-parse', 'HEAD');

(async () => {
  const store = new CheckpointStore(path.join(tmp, 'checkpoints'));
  const cp = await store.create(root, ['a.py', 'new.py'], 'test edit');
  assert.ok(cp.gitRef, 'a git ref was made');
  assert.strictEqual(git('show', `${cp.gitRef}:a.py`), 'x = 2  # uncommitted');
  assert.strictEqual(git('rev-parse', 'HEAD'), headBefore, 'branch untouched');
  assert.strictEqual(git('status', '--porcelain'), statusBefore, 'index and working tree untouched');

  // The "edit": change a.py, create new.py; then restore
  fs.writeFileSync(path.join(root, 'a.py'), 'x = 3\n');
  fs.writeFileSync(path.join(root, 'new.py'), 'y = 1\n');
  assert.strictEqual(store.list()[0].id, cp.id);
  store.restore(store.list()[0]);
  assert.strictEqual(fs.readFileSync(path.join(root, 'a.py'), 'utf8'), 'x = 2  # uncommitted\n');
  assert.ok(!fs.existsSync(path.join(root, 'new.py')), 'created file removed');
  console.log('checkpoint test passed:', cp.gitRef);
})().catch((err) => { console.error(err); process.exit(1); })
  .finally(() => fs.rmSync(tmp, { recursive: true, force: true }));
