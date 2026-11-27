// The interpreter scripts use: $PYTHON, else the sidecar's environment (npm run setup-python),
// else "python" (standard library only: the fallbacks).
const fs = require('fs');
const path = require('path');

const venv = path.join(__dirname, '..', 'sidecar', '.venv');
const own = process.platform === 'win32' ? path.join(venv, 'Scripts', 'python.exe') : path.join(venv, 'bin', 'python');

module.exports = process.env.PYTHON || (fs.existsSync(own) ? own : 'python');
