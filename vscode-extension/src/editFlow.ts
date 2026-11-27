/**
 * Edit -> preview -> approve -> checkpoint -> apply -> verify, looping on what verification
 * finds:
 *
 *  1. The sidecar proposes new file contents (planner -> retriever -> analyst -> editor ->
 *     impact), with every caller of the changed symbols in its context.
 *  2. Each changed file opens as a diff; nothing changes until you approve.
 *  3. A checkpoint is taken (checkpoint.ts), then the change lands as one WorkspaceEdit and is
 *     saved.
 *  4. Verify: errors the language server reports in the changed files and the likely-to-break
 *     dependents (only new ones, compared with before), then the test command if one is set.
 *  5. Problems go back to the sidecar as feedback for another proposal (up to
 *     verify.maxAttempts), which you approve again.
 */
import { exec } from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';
import { CheckpointStore } from './checkpoint';
import { Panel } from './panel';
import { EditResult, ImpactItem, ProposedFile, Sidecar } from './sidecarClient';

export const PROPOSED_SCHEME = 'graphrag-proposed';
const MAX_DIFFS_OPENED = 6;
const MAX_DEPENDENT_FILES_CHECKED = 10;
const TEST_OUTPUT_KEPT = 4000;

/** Serves proposed file contents to the diff editor. */
export class ProposalProvider implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  private changed = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changed.event;

  set(key: string, text: string): vscode.Uri {
    this.contents.set(key, text);
    const uri = vscode.Uri.from({ scheme: PROPOSED_SCHEME, path: `/${key}` });
    this.changed.fire(uri);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.path.slice(1)) ?? '';
  }
}

export interface EditDeps {
  sidecar: () => Promise<Sidecar>;
  root: string;
  panel: () => Panel;
  proposals: ProposalProvider;
  checkpoints: CheckpointStore;
  focus: () => { path: string; line: number } | undefined;
  openFiles: () => string[];
}

const abs = (root: string, rel: string) => path.join(root, ...rel.split('/'));

interface Verification { ok: boolean; feedback: string; summary: string }

function errorKeys(uris: vscode.Uri[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const uri of uris) {
    for (const d of vscode.languages.getDiagnostics(uri)) {
      if (d.severity !== vscode.DiagnosticSeverity.Error) continue;
      const key = `${uri.fsPath}|${d.message}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  return counts;
}

/** Resolves once the language server has gone quiet for these files (or after the timeout). */
function diagnosticsSettled(uris: vscode.Uri[], timeoutMs: number, quietMs = 1500): Promise<void> {
  const watched = new Set(uris.map((u) => u.toString()));
  return new Promise((resolve) => {
    let quiet: NodeJS.Timeout;
    const finish = () => { sub.dispose(); clearTimeout(quiet); clearTimeout(cap); resolve(); };
    const sub = vscode.languages.onDidChangeDiagnostics((e) => {
      if (e.uris.some((u) => watched.has(u.toString()))) {
        clearTimeout(quiet);
        quiet = setTimeout(finish, quietMs);
      }
    });
    quiet = setTimeout(finish, quietMs);
    const cap = setTimeout(finish, timeoutMs);
  });
}

function runTests(command: string, cwd: string): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    exec(command, { cwd, timeout: 10 * 60 * 1000, maxBuffer: 32 * 1024 * 1024 }, (err, stdout, stderr) => {
      const output = `${stdout}\n${stderr}`.trim();
      resolve({ ok: !err, output: output.slice(-TEST_OUTPUT_KEPT) });
    });
  });
}

export class EditFlow {
  constructor(private readonly deps: EditDeps) {}

  async run(instruction: string): Promise<void> {
    const { root, panel } = this.deps;
    const config = vscode.workspace.getConfiguration('graphragCode');
    const maxAttempts = Math.max(1, config.get<number>('verify.maxAttempts', 3));
    const testCommand = config.get<string>('verify.testCommand', '').trim();
    const timeoutMs = 1000 * config.get<number>('verify.diagnosticsTimeoutSeconds', 8);

    let feedback = '';
    let targets: string[] = [];
    const focus = this.deps.focus();
    // Tests that already fail before any change can't tell whether the change works
    let testsUsable = Boolean(testCommand);
    let baselineChecked = false;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      panel().reset('edit', attempt === 1 ? `Edit: ${instruction}` : `Edit (attempt ${attempt}): ${instruction}`);
      const sidecar = await this.deps.sidecar();
      const result = await sidecar.request<EditResult>('edit', {
        instruction, focus, openFiles: this.deps.openFiles(), targets, feedback: feedback || undefined,
      }, (e) => panel().event(e));
      panel().result('edit', result);
      if (!result.files.length) {
        panel().note('error', `No change to apply. ${result.skipped.join('; ') || 'The model returned no edit blocks.'}`);
        return;
      }

      const approved = await this.preview(result);
      if (!approved) {
        panel().note('info', 'Not applied. Nothing was changed.');
        return;
      }

      const impactedUris = this.dependentFiles(result.impact, result.files);
      const changedUris = result.files.map((f) => vscode.Uri.file(abs(root, f.path)));
      // Errors that were there before the change don't count against it
      await Promise.all(impactedUris.concat(changedUris.filter((u, i) => !result.files[i].created))
        .map((u) => vscode.workspace.openTextDocument(u).then(undefined, () => undefined)));
      await diagnosticsSettled(impactedUris.concat(changedUris), Math.min(timeoutMs, 4000), 800);
      const before = errorKeys(impactedUris.concat(changedUris));
      if (testsUsable && !baselineChecked) {
        baselineChecked = true;
        panel().note('info', `Running \`${testCommand}\` before the change, to know what already fails...`);
        const baseline = await runTests(testCommand, root);
        if (!baseline.ok) {
          testsUsable = false;
          panel().note('warning', 'The tests already fail without this change, so they are not used to verify it.');
        }
      }

      const checkpoint = await this.deps.checkpoints.create(root, result.files.map((f) => f.path), instruction);
      await this.apply(result.files);
      await sidecar.request('update', { paths: result.files.map((f) => f.path) });
      panel().note('success', `Applied to ${result.files.length} file(s). Checkpoint ${checkpoint.id}`
        + `${checkpoint.gitRef ? ` (\`${checkpoint.gitRef}\`)` : ''}: "GraphRAG Code: Restore a Checkpoint" undoes it.`);

      const verdict = await this.verify(changedUris, impactedUris, before, timeoutMs, testsUsable ? testCommand : '');
      panel().note(verdict.ok ? 'success' : 'warning', verdict.summary);
      if (verdict.ok) return;
      if (attempt === maxAttempts) {
        const choice = await vscode.window.showWarningMessage(
          `The edit still has problems after ${maxAttempts} attempt(s).`, 'Keep it', 'Restore checkpoint');
        if (choice === 'Restore checkpoint') {
          const paths = this.deps.checkpoints.restore(checkpoint);
          await sidecar.request('update', { paths });
          panel().note('info', 'Restored the files from the checkpoint.');
        }
        return;
      }
      feedback = verdict.feedback;
      targets = result.changedSymbols;
      panel().note('info', 'Sending the problems back to the model for another attempt...');
    }
  }

  /** Opens a diff per changed file and asks for approval (modal). */
  private async preview(result: EditResult): Promise<boolean> {
    const { root, proposals } = this.deps;
    const stamp = Date.now().toString(36);
    for (const f of result.files.slice(0, MAX_DIFFS_OPENED)) {
      const right = proposals.set(`${stamp}/${f.path}`, f.proposed);
      const left = f.created ? proposals.set(`${stamp}/empty/${f.path}`, '') : vscode.Uri.file(abs(root, f.path));
      await vscode.commands.executeCommand('vscode.diff', left, right,
        `${path.basename(f.path)}: proposed${f.created ? ' (new file)' : ''}`, { preview: false, preserveFocus: true });
    }
    const direct = result.impact.filter((a) => a.direct).map((a) => a.qualname);
    const lines = [
      result.summary,
      `Files: ${result.files.map((f) => f.path).join(', ')}`,
      direct.length ? `Likely to break: ${direct.slice(0, 6).join(', ')}${direct.length > 6 ? ', ...' : ''}` : '',
      result.syntaxErrors.length ? `Warning, still doesn't parse: ${result.syntaxErrors.join('; ')}` : '',
      result.skipped.length ? `Skipped: ${result.skipped.join('; ')}` : '',
      result.files.length > MAX_DIFFS_OPENED ? `(${result.files.length - MAX_DIFFS_OPENED} more file(s) not shown as diffs)` : '',
      'A checkpoint is taken before anything changes.',
    ].filter(Boolean);
    const choice = await vscode.window.showInformationMessage(
      `Apply the change to ${result.files.length} file(s)?`, { modal: true, detail: lines.join('\n') }, 'Apply');
    return choice === 'Apply';
  }

  private async apply(files: ProposedFile[]): Promise<void> {
    const { root } = this.deps;
    const edit = new vscode.WorkspaceEdit();
    for (const f of files) {
      const uri = vscode.Uri.file(abs(root, f.path));
      if (f.created) {
        edit.createFile(uri, { ignoreIfExists: false });
        edit.insert(uri, new vscode.Position(0, 0), f.proposed);
      } else {
        const doc = await vscode.workspace.openTextDocument(uri);
        const all = new vscode.Range(doc.positionAt(0), doc.positionAt(doc.getText().length));
        edit.replace(uri, all, f.proposed);
      }
    }
    if (!(await vscode.workspace.applyEdit(edit))) throw new Error('VS Code did not apply the edit.');
    for (const f of files) {
      const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(abs(root, f.path)));
      await doc.save();
    }
  }

  private dependentFiles(impact: ImpactItem[], files: ProposedFile[]): vscode.Uri[] {
    const changed = new Set(files.map((f) => f.path));
    const paths = [...new Set(impact.map((a) => a.path))].filter((p) => !changed.has(p));
    return paths.slice(0, MAX_DEPENDENT_FILES_CHECKED).map((p) => vscode.Uri.file(abs(this.deps.root, p)));
  }

  private async verify(changed: vscode.Uri[], dependents: vscode.Uri[], before: Map<string, number>,
    timeoutMs: number, testCommand: string): Promise<Verification> {
    const { panel, root } = this.deps;
    panel().note('info', 'Verifying: waiting for the language server...');
    const all = changed.concat(dependents);
    await diagnosticsSettled(all, timeoutMs);
    const problems: string[] = [];
    for (const uri of all) {
      for (const d of vscode.languages.getDiagnostics(uri)) {
        if (d.severity !== vscode.DiagnosticSeverity.Error) continue;
        const key = `${uri.fsPath}|${d.message}`;
        const seen = before.get(key) || 0;
        if (seen > 0) {
          before.set(key, seen - 1); // was there before the change
          continue;
        }
        problems.push(`${path.relative(root, uri.fsPath).split(path.sep).join('/')}:${d.range.start.line + 1}: ${d.message}`);
      }
    }
    let testReport = '';
    if (testCommand) {
      panel().note('info', `Running \`${testCommand}\`...`);
      const tests = await runTests(testCommand, root);
      if (!tests.ok) testReport = tests.output;
    }
    if (!problems.length && !testReport) {
      return { ok: true, feedback: '', summary: `Verified: no new errors${testCommand ? ' and the tests pass' : ''}.` };
    }
    const feedback = [
      problems.length ? `New errors reported by the language server:\n${problems.slice(0, 30).join('\n')}` : '',
      testReport ? `The test command \`${testCommand}\` failed:\n${testReport}` : '',
    ].filter(Boolean).join('\n\n');
    const summary = [
      problems.length ? `${problems.length} new error(s): ${problems.slice(0, 3).join('; ')}${problems.length > 3 ? '; ...' : ''}` : '',
      testReport ? 'the tests fail' : '',
    ].filter(Boolean).join(', and ');
    return { ok: false, feedback, summary: `Verification found problems: ${summary}.` };
  }
}
