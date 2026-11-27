// Runs the Python tests of the shared core and the sidecar (interpreter: scripts/python.js).
// With plain "python" (PYTHON=python) the tree-sitter and LanceDB tests are skipped and the
// fallbacks are tested instead.
const { spawnSync } = require('child_process');
const path = require('path');

const python = require('./python');
console.log(`Python: ${python}`);
const repo = path.join(__dirname, '..', '..');
const core = path.join(repo, 'core');
const sidecar = path.join(repo, 'vscode-extension', 'sidecar');

function run(cwd, extraPath) {
  const env = { ...process.env, PYTHONPATH: [extraPath, core].join(path.delimiter) };
  const r = spawnSync(python, ['-m', 'unittest', 'discover', '-s', 'tests'], { cwd, env, stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status || 1);
}

run(core, core);
run(sidecar, sidecar);
