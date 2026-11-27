// Makes the sidecar's own Python environment (sidecar/.venv) with sidecar/requirements.txt,
// then checks tree-sitter and LanceDB load (Windows Smart App Control can block a package's
// compiled modules). The extension uses this environment when graphragCode.pythonPath is left
// at "python". Usage: npm run setup-python [-- <python to create it with>]
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const sidecar = path.join(__dirname, '..', 'sidecar');
const venv = path.join(sidecar, '.venv');
const base = process.argv[2] || process.env.PYTHON || 'python';
const python = process.platform === 'win32' ? path.join(venv, 'Scripts', 'python.exe') : path.join(venv, 'bin', 'python');

function run(cmd, args) {
  console.log(`> ${cmd} ${args.join(' ')}`);
  const r = spawnSync(cmd, args, { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status || 1);
}

if (!fs.existsSync(python)) run(base, ['-m', 'venv', venv]);
run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', '-r', path.join(sidecar, 'requirements.txt')]);
run(python, ['-c', [
  'import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]',
  'from graphrag_code import parse_treesitter as ts, vectors as v',
  'print("tree-sitter:", "ok" if ts.available() else "NOT AVAILABLE - " + ts.IMPORT_ERROR)',
  'print("LanceDB:", "ok" if v.lancedb is not None else "NOT AVAILABLE - " + v.LANCE_ERROR)',
].join('\n'), sidecar, path.join(__dirname, '..', '..', 'core')]);
console.log(`Sidecar environment ready: ${python}`);
