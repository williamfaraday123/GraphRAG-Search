// Copies the repo's shared core (../core/graphrag_core) into sidecar/vendor, so a packaged
// extension (.vsix) carries it. Run by `vscode:prepublish`; from a checkout the extension
// uses ../core directly.
const fs = require('fs');
const path = require('path');

const source = path.join(__dirname, '..', '..', 'core', 'graphrag_core');
const target = path.join(__dirname, '..', 'sidecar', 'vendor', 'graphrag_core');

if (!fs.existsSync(source)) {
  console.error(`Shared core not found at ${source}`);
  process.exit(1);
}
fs.rmSync(target, { recursive: true, force: true });
fs.cpSync(source, target, { recursive: true, filter: (p) => !p.includes('__pycache__') });
console.log(`Copied graphrag_core to ${path.relative(process.cwd(), target)}`);
