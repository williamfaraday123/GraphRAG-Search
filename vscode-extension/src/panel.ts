/**
 * The "GraphRAG Code" panel: one run at a time (a question, an edit, a graph or an impact
 * view). It shows the workflow's progress, the streamed answer, the context ranked by PPR and
 * similarity, and the code graph around it (media/panel.js draws it with Cytoscape). Clicking
 * a symbol anywhere in it opens the code.
 */
import * as path from 'path';
import * as vscode from 'vscode';
import { SidecarEvent } from './sidecarClient';

export type PanelMessage =
  | { type: 'reset'; kind: string; title: string }
  | { type: 'event'; event: SidecarEvent }
  | { type: 'result'; kind: string; result: unknown }
  | { type: 'note'; level: 'info' | 'warning' | 'error' | 'success'; text: string };

export class Panel {
  private static current: Panel | undefined;
  private queue: PanelMessage[] = [];
  private ready = false;

  /** The open panel, if any (without opening one). */
  static existing(): Panel | undefined {
    return Panel.current;
  }

  static show(extensionUri: vscode.Uri, root: () => string | undefined): Panel {
    if (Panel.current) {
      Panel.current.panel.reveal(vscode.ViewColumn.Beside, true);
      return Panel.current;
    }
    Panel.current = new Panel(extensionUri, root);
    return Panel.current;
  }

  private readonly panel: vscode.WebviewPanel;

  private constructor(private readonly extensionUri: vscode.Uri, private readonly root: () => string | undefined) {
    const cytoscapeDir = vscode.Uri.joinPath(extensionUri, 'node_modules', 'cytoscape', 'dist');
    this.panel = vscode.window.createWebviewPanel('graphragCode', 'GraphRAG Code',
      { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true }, {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media'), cytoscapeDir],
      });
    this.panel.webview.html = this.html(cytoscapeDir);
    this.panel.onDidDispose(() => { Panel.current = undefined; });
    this.panel.webview.onDidReceiveMessage((msg) => this.onMessage(msg));
  }

  post(message: PanelMessage): void {
    if (!this.ready) {
      this.queue.push(message);
      return;
    }
    void this.panel.webview.postMessage(message);
  }

  reset(kind: string, title: string): void {
    this.post({ type: 'reset', kind, title });
  }

  event(event: SidecarEvent): void {
    this.post({ type: 'event', event });
  }

  result(kind: string, result: unknown): void {
    this.post({ type: 'result', kind, result });
  }

  note(level: 'info' | 'warning' | 'error' | 'success', text: string): void {
    this.post({ type: 'note', level, text });
  }

  private async onMessage(msg: any): Promise<void> {
    if (msg.type === 'ready') {
      this.ready = true;
      this.queue.splice(0).forEach((m) => void this.panel.webview.postMessage(m));
      return;
    }
    if (msg.type === 'open' && typeof msg.path === 'string') {
      const root = this.root();
      const file = path.isAbsolute(msg.path) || !root ? msg.path : path.join(root, ...msg.path.split('/'));
      const line = Math.max(0, (Number(msg.line) || 1) - 1);
      if (msg.path.endsWith('/')) { // a folder node: show it in the Explorer
        await vscode.commands.executeCommand('revealInExplorer', vscode.Uri.file(file));
        return;
      }
      try {
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
        const pos = new vscode.Position(line, 0);
        await vscode.window.showTextDocument(doc, {
          viewColumn: vscode.ViewColumn.One, selection: new vscode.Range(pos, pos), preserveFocus: false,
        });
      } catch (err: any) {
        void vscode.window.showWarningMessage(`Could not open ${msg.path}: ${err.message}`);
      }
    }
  }

  private html(cytoscapeDir: vscode.Uri): string {
    const webview = this.panel.webview;
    const media = (name: string) => webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', name));
    const cytoscape = webview.asWebviewUri(vscode.Uri.joinPath(cytoscapeDir, 'cytoscape.min.js'));
    const nonce = Array.from({ length: 32 }, () => Math.floor(Math.random() * 36).toString(36)).join('');
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${media('panel.css')}">
<title>GraphRAG Code</title>
</head>
<body>
  <header>
    <h1 id="title">GraphRAG Code</h1>
    <p id="status" class="status" role="status"></p>
  </header>
  <ol id="steps" class="steps" aria-label="Workflow steps"></ol>
  <div id="notes" aria-live="polite"></div>
  <section id="answer-section" hidden>
    <h2 id="answer-heading">Answer</h2>
    <div id="answer" class="answer"></div>
  </section>
  <section id="graph-answer-section" hidden>
    <h2>From the code graph</h2>
    <p id="graph-answer-summary" class="hint"></p>
    <ul id="graph-answer" class="symbols"></ul>
  </section>
  <section id="relations-section" hidden>
    <h2>Relationships the model read</h2>
    <p class="hint">Edges of the code graph (static analysis) given to the model beside the code.</p>
    <ul id="folder-links" class="symbols"></ul>
    <ul id="relations" class="symbols"></ul>
  </section>
  <section id="impact-section" hidden>
    <h2>Likely to break</h2>
    <p class="hint">Code that depends on what changes: PPR pushed backwards along calls, subclassing, type use and imports. Direct dependents first.</p>
    <ul id="impact" class="symbols"></ul>
  </section>
  <section id="graph-section" hidden>
    <h2>Code graph</h2>
    <p class="hint">Colour and size show the PPR score from the question, cursor or change. Ringed: what the request is about. Click a node to open it.</p>
    <div id="legend" class="legend"></div>
    <div id="graph" class="graph" aria-label="Code graph"></div>
  </section>
  <section id="context-section" hidden>
    <h2 id="context-heading">Context</h2>
    <ul id="context" class="symbols"></ul>
  </section>
  <script nonce="${nonce}" src="${cytoscape}"></script>
  <script nonce="${nonce}" src="${media('panel.js')}"></script>
</body>
</html>`;
  }
}
