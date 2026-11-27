/**
 * Checkpoints taken before an edit is applied, so it can be undone.
 *
 * Each one keeps the affected files' exact contents (in the extension's storage; files the
 * edit creates are recorded as not existing, so restoring deletes them). In a git repository
 * it is also a real commit: HEAD's tree plus the files as they are now, built with a temporary
 * index and kept under refs/graphrag/checkpoints/<id>. That never touches your branch, index
 * or working tree; `git show refs/graphrag/checkpoints/<id>` shows it.
 */
import { execFile } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export interface CheckpointFile { path: string; existed: boolean; content: string }
export interface Checkpoint { id: string; label: string; created: number; root: string; files: CheckpointFile[]; gitRef?: string }

function git(cwd: string, args: string[], env: NodeJS.ProcessEnv = {}): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile('git', args, { cwd, env: { ...process.env, ...env }, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) reject(new Error(stderr.trim() || err.message));
      else resolve(stdout.trim());
    });
  });
}

const absolute = (root: string, rel: string) => path.join(root, ...rel.split('/'));

async function gitCheckpoint(root: string, id: string, label: string, existing: string[]): Promise<string | undefined> {
  try {
    if ((await git(root, ['rev-parse', '--is-inside-work-tree'])) !== 'true') return undefined;
    const top = await git(root, ['rev-parse', '--show-toplevel']);
    const head = await git(top, ['rev-parse', '--verify', '-q', 'HEAD']).catch(() => '');
    const index = path.join(os.tmpdir(), `graphrag-index-${id}`);
    const env = { GIT_INDEX_FILE: index };
    try {
      if (head) await git(top, ['read-tree', head], env);
      const files = existing.map((rel) => path.relative(top, absolute(root, rel)));
      if (files.length) await git(top, ['add', '-f', '--', ...files], env);
      const tree = await git(top, ['write-tree'], env);
      const commit = await git(top, ['commit-tree', tree, ...(head ? ['-p', head] : []), '-m', `GraphRAG Code checkpoint: ${label}`]);
      const ref = `refs/graphrag/checkpoints/${id}`;
      await git(top, ['update-ref', ref, commit]);
      return ref;
    } finally {
      fs.rmSync(index, { force: true });
    }
  } catch {
    return undefined; // not a git repository, or git missing: the stored copies still restore
  }
}

export class CheckpointStore {
  constructor(private readonly dir: string) {}

  async create(root: string, relPaths: string[], label: string): Promise<Checkpoint> {
    const id = `${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}-${Math.random().toString(36).slice(2, 6)}`;
    const files: CheckpointFile[] = relPaths.map((rel) => {
      const full = absolute(root, rel);
      const existed = fs.existsSync(full);
      return { path: rel, existed, content: existed ? fs.readFileSync(full, 'utf8') : '' };
    });
    const gitRef = await gitCheckpoint(root, id, label, files.filter((f) => f.existed).map((f) => f.path));
    const checkpoint: Checkpoint = { id, label, created: Date.now(), root, files, gitRef };
    fs.mkdirSync(this.dir, { recursive: true });
    fs.writeFileSync(path.join(this.dir, `${id}.json`), JSON.stringify(checkpoint));
    return checkpoint;
  }

  list(): Checkpoint[] {
    if (!fs.existsSync(this.dir)) return [];
    return fs.readdirSync(this.dir)
      .filter((f) => f.endsWith('.json'))
      .map((f) => JSON.parse(fs.readFileSync(path.join(this.dir, f), 'utf8')) as Checkpoint)
      .sort((a, b) => b.created - a.created);
  }

  /** Puts the files back as they were; returns the paths it changed. */
  restore(checkpoint: Checkpoint): string[] {
    for (const f of checkpoint.files) {
      const full = absolute(checkpoint.root, f.path);
      if (f.existed) {
        fs.mkdirSync(path.dirname(full), { recursive: true });
        fs.writeFileSync(full, f.content, 'utf8');
      } else {
        fs.rmSync(full, { force: true });
      }
    }
    return checkpoint.files.map((f) => f.path);
  }
}
