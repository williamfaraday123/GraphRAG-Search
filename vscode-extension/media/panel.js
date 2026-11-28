// The GraphRAG Code panel (webview side). Receives runs from src/panel.ts; draws the code
// graph with Cytoscape. Plain script: no build step.
(function () {
  const vscode = acquireVsCodeApi();
  const $ = (id) => document.getElementById(id);
  let cy = null;
  let answerText = '';

  const EDGE_STYLES = {
    calls: { style: 'solid', arrow: 'triangle', label: 'calls' },
    renders: { style: 'solid', arrow: 'diamond', label: 'renders' },
    extends: { style: 'dashed', arrow: 'triangle-backcurve', label: 'extends' },
    implements: { style: 'dashed', arrow: 'triangle-tee', label: 'implements' },
    uses_type: { style: 'dashed', arrow: 'vee', label: 'uses type' },
    uses: { style: 'dashed', arrow: 'triangle', label: 'uses (references)' },
    contains: { style: 'dotted', arrow: 'none', label: 'contains' },
    contains_dir: { style: 'dotted', arrow: 'none', label: 'in folder' },
    imports: { style: 'dotted', arrow: 'vee', label: 'imports' },
  };
  const KIND_SHAPES = {
    class: 'round-rectangle', method: 'ellipse', function: 'ellipse', module: 'rectangle',
    component: 'hexagon', interface: 'round-diamond', type: 'round-diamond', folder: 'barrel',
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function open(path, line) {
    vscode.postMessage({ type: 'open', path, line });
  }

  // `path/to/file.py:42` (in backticks) becomes a link to that line
  const CODE_REF = /^([\w./\\@-]+\.(?:py|tsx?|mts|cts|jsx?|mjs|cjs))(?::(\d+))?(?:-\d+)?$/;

  function inline(text) {
    return escapeHtml(text)
      .replace(/`([^`]+)`/g, (m, code) => {
        const ref = code.match(CODE_REF);
        if (ref) return `<a href="#" class="code-ref" data-path="${ref[1]}" data-line="${ref[2] || 1}"><code>${code}</code></a>`;
        return `<code>${code}</code>`;
      })
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  }

  // Enough Markdown for model answers: fenced code, headings, lists, paragraphs
  function markdown(text) {
    const out = [];
    const lines = text.split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        const code = [];
        i += 1;
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        i += 1;
        out.push(`<pre><code>${escapeHtml(code.join('\n'))}</code></pre>`);
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        out.push(`<h${heading[1].length + 2}>${inline(heading[2])}</h${heading[1].length + 2}>`);
      } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''))}</li>`);
          i += 1;
        }
        out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
        continue;
      } else if (line.trim()) {
        out.push(`<p>${inline(line)}</p>`);
      }
      i += 1;
    }
    return out.join('');
  }

  function reset(kind, title) {
    $('title').textContent = title;
    $('status').textContent = '';
    $('steps').innerHTML = '';
    $('notes').innerHTML = '';
    $('answer').innerHTML = '';
    $('impact').innerHTML = '';
    $('context').innerHTML = '';
    answerText = '';
    $('answer-heading').textContent = kind === 'edit' ? 'Proposed change' : 'Answer';
    ['answer-section', 'graph-answer-section', 'relations-section', 'impact-section', 'graph-section', 'context-section',
      'arch-section', 'components-section'].forEach((id) => { $(id).hidden = true; });
    ['graph-answer', 'folder-links', 'relations', 'components', 'arch-details'].forEach((id) => { $(id).innerHTML = ''; });
    if (cy) { cy.destroy(); cy = null; }
    if (archCy) { archCy.destroy(); archCy = null; }
    archStack = [];
  }

  function onEvent(e) {
    if (e.type === 'status') $('status').textContent = e.message || e.stage || '';
    if (e.type === 'agent_step') {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${escapeHtml(e.agent)}</strong> ${escapeHtml(e.detail || '')}`;
      $('steps').appendChild(li);
    }
    if (e.type === 'chunk') {
      answerText += e.content || '';
      $('answer-section').hidden = false;
      $('answer').innerHTML = markdown(answerText);
    }
  }

  function note(level, text) {
    const p = document.createElement('p');
    p.className = `note note-${level}`;
    p.innerHTML = inline(text).replace(/\n/g, '<br>');
    $('notes').appendChild(p);
  }

  function symbolList(el, items, describe) {
    el.innerHTML = '';
    items.forEach((it) => {
      const li = document.createElement('li');
      li.innerHTML = `<a href="#" class="code-ref" data-path="${escapeHtml(it.path)}" data-line="${it.start || it.line || 1}">`
        + `<code>${escapeHtml(it.qualname || it.path)}</code></a> <span class="where">${escapeHtml(it.path)}:${it.start || it.line}</span>`
        + `<div class="why">${describe(it)}</div>`;
      el.appendChild(li);
    });
  }

  const pct = (x) => `${Math.round((x || 0) * 100)}%`;

  function showContext(context, heading) {
    if (!context || !context.length) return;
    $('context-section').hidden = false;
    $('context-heading').textContent = heading || `Context (${context.length} symbols)`;
    symbolList($('context'), context, (c) => `${escapeHtml(c.reason)} · score ${pct(c.score)} · similarity ${pct(c.semantic)} · PPR ${pct(c.ppr)}`);
  }

  function showImpact(impact) {
    $('impact-section').hidden = false;
    if (!impact.length) {
      $('impact').innerHTML = '<li class="why">Nothing in the index depends on it.</li>';
      return;
    }
    symbolList($('impact'), impact, (a) => (a.via
      ? `in ${escapeHtml(a.component || 'another component')}, across the boundary: ${escapeHtml(a.via)}`
      : `${a.direct ? 'direct dependent' : 'depends on it indirectly'} · PPR ${a.score.toFixed(3)}`));
  }

  // One edge of the graph as a line; the code it starts from opens on click
  function relationItem(r) {
    const li = document.createElement('li');
    const where = r.path ? ` <a href="#" class="code-ref" data-path="${escapeHtml(r.path)}" data-line="${r.line || 1}">open</a>` : '';
    li.innerHTML = `<code>${escapeHtml(r.text)}</code>${where}`;
    return li;
  }

  function showGraphAnswer(g) {
    if (!g) return;
    $('graph-answer-section').hidden = false;
    $('graph-answer-summary').textContent = g.summary;
    g.facts.forEach((f) => $('graph-answer').appendChild(relationItem(f)));
  }

  function showRelations(relations, folderLinks) {
    if (!(relations && relations.length) && !(folderLinks && folderLinks.length)) return;
    $('relations-section').hidden = false;
    (folderLinks || []).forEach((l) => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${escapeHtml(l.text)}</strong>`;
      $('folder-links').appendChild(li);
    });
    (relations || []).forEach((r) => $('relations').appendChild(relationItem(r)));
  }

  function colourFor(score) {
    // Low PPR: the editor's muted colour; high: the theme's accent
    const t = Math.max(0, Math.min(1, Math.sqrt(score)));
    return `hsl(${210 - t * 170}, ${35 + t * 50}%, ${62 - t * 17}%)`;
  }

  function showGraph(graph) {
    if (!graph || !graph.nodes.length) {
      if (graph) note('warning', 'No indexed symbols were found around this cursor.');
      return;
    }
    if (typeof cytoscape === 'undefined') {
      note('warning', 'The code graph library could not be loaded.');
      return;
    }
    $('graph-section').hidden = false;
    const kinds = new Set(graph.edges.map((e) => e.kind));
    $('legend').innerHTML = [...kinds].map((k) => `<span class="edge-${k}">${EDGE_STYLES[k] ? EDGE_STYLES[k].label : k}</span>`).join('');
    const fg = getComputedStyle(document.body).getPropertyValue('--vscode-editor-foreground') || '#ccc';
    const elements = [
      ...graph.nodes.map((n) => ({ data: { ...n, colour: colourFor(n.score), size: 18 + 34 * Math.sqrt(n.score) } })),
      ...graph.edges.map((e, i) => ({ data: { id: `e${i}`, ...e } })),
    ];
    if (cy) cy.destroy();
    cy = cytoscape({
      container: $('graph'),
      elements,
      wheelSensitivity: 0.3,
      style: [
        { selector: 'node', style: {
          'background-color': 'data(colour)', width: 'data(size)', height: 'data(size)',
          label: 'data(label)', color: fg, 'font-size': 10, 'text-valign': 'bottom', 'text-margin-y': 3,
          'text-wrap': 'ellipsis', 'text-max-width': 140,
        } },
        ...Object.entries(KIND_SHAPES).map(([k, shape]) => ({ selector: `node[kind = "${k}"]`, style: { shape } })),
        { selector: 'node[?highlight]', style: { 'border-width': 3, 'border-color': fg } },
        { selector: 'edge', style: {
          width: 1.4, 'curve-style': 'bezier', 'line-color': '#888', 'target-arrow-color': '#888', opacity: 0.7,
        } },
        ...Object.entries(EDGE_STYLES).map(([k, s]) => ({ selector: `edge[kind = "${k}"]`, style: {
          'line-style': s.style, 'target-arrow-shape': s.arrow,
        } })),
        { selector: 'edge[?guessed]', style: { opacity: 0.35 } },
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: 9000, idealEdgeLength: 70, padding: 20 },
    });
    cy.on('tap', 'node', (evt) => open(evt.target.data('path'), evt.target.data('line')));
  }

  // --- architecture --------------------------------------------------------------------------
  // Connection kinds: colour, line, width ('count': by how many code references)
  const ARCH_EDGES = {
    http: { color: '#4f8fea', style: 'solid', width: 3, label: 'HTTP' },
    grpc: { color: '#9a6ff0', style: 'solid', width: 3, label: 'gRPC' },
    process: { color: '#e0a030', style: 'dashed', width: 2.5, label: 'starts a process' },
    code: { color: '#8a8a8a', style: 'solid', width: 'count', label: 'code references' },
    datastore: { color: '#2fb383', style: 'solid', width: 2, label: 'datastore' },
    'model API': { color: '#e05aa0', style: 'dashed', width: 2, label: 'model API' },
    compose: { color: '#9a9a9a', style: 'dotted', width: 1.2, label: 'docker-compose' },
  };
  const ARCH_SHAPES = {
    code: 'round-rectangle', cluster: 'round-rectangle', module: 'rectangle', datastore: 'barrel', queue: 'tag', service: 'ellipse',
  };
  const INFRA = new Set(['datastore', 'queue', 'service']);
  let archCy = null;
  let archStack = [];

  function archEdgeWidth(e) {
    const s = ARCH_EDGES[e.kind];
    if (!s) return 1.5;
    return s.width === 'count' ? 1 + Math.min(5, Math.log2(1 + (e.count || 1))) : s.width;
  }

  function showArchitecture(view) {
    if (!view) return;
    $('arch-section').hidden = false;
    const at = archStack.findIndex((v) => v.node === view.node);
    if (view.level === 'system') archStack = [view];
    else if (at >= 0) archStack = archStack.slice(0, at).concat([view]);
    else archStack.push(view);
    renderArchitecture();
  }

  function renderArchitecture() {
    const view = archStack[archStack.length - 1];
    $('arch-crumbs').innerHTML = '';
    archStack.forEach((v, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = v.title;
      b.disabled = i === archStack.length - 1;
      b.addEventListener('click', () => { archStack = archStack.slice(0, i + 1); renderArchitecture(); });
      $('arch-crumbs').appendChild(b);
      if (i < archStack.length - 1) $('arch-crumbs').appendChild(document.createTextNode(' › '));
    });
    const how = { manifests: 'build files and docker-compose services', clusters: 'Leiden clustering of the module graph', folders: 'top-level folders' }[view.method] || view.method;
    $('arch-hint').textContent = (view.level === 'system'
      ? `Components from ${how}. Connections: code references (static analysis) and detected runtime links. Click a node or an edge for details and evidence.`
      : view.grouped
        ? 'Leiden clusters of this component\'s modules (sub-components). Faded nodes are other components.'
        : 'Modules of this part. Faded nodes are other components.') + (view.notes && view.notes.length ? ` ${view.notes.join(' ')}` : '');
    $('arch-details').innerHTML = '';
    if (typeof cytoscape === 'undefined') {
      note('warning', 'The graph library could not be loaded.');
      return;
    }
    const showCompose = $('arch-compose').checked;
    const showInfra = $('arch-infra').checked;
    const nodes = view.nodes.filter((n) => showInfra || !INFRA.has(n.kind));
    const keep = new Set(nodes.map((n) => n.id));
    const edges = view.edges.filter((e) => keep.has(e.source) && keep.has(e.target) && (showCompose || e.kind !== 'compose'));
    const kinds = [...new Set(edges.map((e) => e.kind))];
    $('arch-legend').innerHTML = kinds.map((k) => {
      const s = ARCH_EDGES[k] || { color: '#888', style: 'solid', label: k };
      return `<span style="border-bottom: 2px ${s.style} ${s.color}">${escapeHtml(s.label)}</span>`;
    }).join('');
    const fg = getComputedStyle(document.body).getPropertyValue('--vscode-editor-foreground') || '#ccc';
    const bg = getComputedStyle(document.body).getPropertyValue('--vscode-editor-background') || '#1e1e1e';
    const display = (n) => {
      const bits = [];
      if (n.languages && n.languages.length) bits.push(n.languages.slice(0, 2).join('/'));
      if (n.files && n.kind !== 'module') bits.push(`${n.files} file${n.files === 1 ? '' : 's'}`);
      return bits.length ? `${n.label}\n${bits.join(' · ')}` : n.label;
    };
    if (archCy) archCy.destroy();
    archCy = cytoscape({
      container: $('arch'),
      elements: [
        ...nodes.map((n) => ({ data: { ...n, display: display(n) } })),
        ...edges.map((e, i) => ({ data: { ...e, id: `a${i}`, width: archEdgeWidth(e) } })),
      ],
      wheelSensitivity: 0.3,
      style: [
        { selector: 'node', style: {
          label: 'data(display)', color: fg, 'font-size': 11, 'text-valign': 'center', 'text-halign': 'center',
          'text-wrap': 'wrap', 'text-max-width': 170, width: 'label', height: 'label', padding: 10,
          'background-color': '#3a6ea5', 'background-opacity': 0.25, 'border-width': 1.5, 'border-color': '#3a6ea5',
          shape: 'round-rectangle',
        } },
        ...Object.entries(ARCH_SHAPES).map(([k, shape]) => ({ selector: `node[kind = "${k}"]`, style: { shape } })),
        { selector: 'node[kind = "datastore"], node[kind = "queue"]', style: { 'background-color': '#2fb383', 'border-color': '#2fb383' } },
        { selector: 'node[kind = "service"]', style: { 'background-color': '#e05aa0', 'border-color': '#e05aa0' } },
        { selector: 'node[kind = "cluster"]', style: { 'background-color': '#9a6ff0', 'border-color': '#9a6ff0' } },
        { selector: 'node[kind = "module"]', style: { 'background-color': '#8a8a8a', 'border-color': '#8a8a8a', 'font-size': 10 } },
        { selector: 'node[?outside]', style: { opacity: 0.55, 'border-style': 'dashed' } },
        { selector: 'node[?drill]', style: { 'border-width': 2.5 } },
        { selector: 'node:selected', style: { 'border-color': fg, 'border-width': 3 } },
        { selector: 'edge', style: {
          width: 'data(width)', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.9,
          label: 'data(label)', 'font-size': 8, color: fg, 'text-rotation': 'autorotate',
          'text-background-color': bg, 'text-background-opacity': 0.8, 'text-background-padding': 1,
          'text-wrap': 'ellipsis', 'text-max-width': 160,
        } },
        ...Object.entries(ARCH_EDGES).map(([k, s]) => ({ selector: `edge[kind = "${k}"]`, style: {
          'line-color': s.color, 'target-arrow-color': s.color, 'line-style': s.style,
        } })),
        { selector: 'edge:selected', style: { 'line-color': fg, 'target-arrow-color': fg } },
      ],
      layout: edges.length
        ? { name: 'breadthfirst', directed: true, spacingFactor: 1.15, avoidOverlap: true, padding: 20 }
        : { name: 'grid', padding: 20 },
    });
    archCy.on('tap', 'node', (evt) => archNodeDetails(evt.target.data()));
    archCy.on('tap', 'edge', (evt) => archEdgeDetails(evt.target.data()));
    archCy.on('dbltap', 'node', (evt) => {
      const n = evt.target.data();
      if (n.drill) vscode.postMessage({ type: 'drill', node: n.id });
    });
  }

  function evidenceList(evidence) {
    return `<ul class="symbols">${(evidence || []).map((ev) => `<li>${ev.line
      ? `<a href="#" class="code-ref" data-path="${escapeHtml(ev.path)}" data-line="${ev.line}"><code>${escapeHtml(ev.path)}:${ev.line}</code></a>`
      : `<code>${escapeHtml(ev.path)}</code>`} <span class="why">${escapeHtml(ev.text || '')}</span></li>`).join('')}</ul>`;
  }

  function archNodeDetails(n) {
    const bits = [n.kind === 'code' ? 'component' : n.kind, n.source, (n.languages || []).join('/'),
      n.files ? `${n.files} files` : '', n.symbols ? `${n.symbols} symbols` : ''].filter(Boolean);
    const actions = [];
    if (n.drill) actions.push(`<button type="button" data-act="drill">Open inside</button>`);
    if (n.kind === 'module') actions.push(`<button type="button" data-act="symbols">Show its code graph</button>`, `<button type="button" data-act="open">Open file</button>`);
    if (n.path && n.path.endsWith('/')) actions.push(`<button type="button" data-act="reveal">Reveal folder</button>`);
    $('arch-details').innerHTML = `<h3>${escapeHtml(n.label)}</h3><p class="hint">${escapeHtml(bits.join(' · '))}</p>`
      + (n.description ? `<p>${escapeHtml(n.description)}</p>` : '')
      + `<div class="arch-actions">${actions.join(' ')}</div>`;
    $('arch-details').querySelectorAll('button[data-act]').forEach((b) => b.addEventListener('click', () => {
      const act = b.dataset.act;
      if (act === 'drill') vscode.postMessage({ type: 'drill', node: n.id });
      else if (act === 'symbols') vscode.postMessage({ type: 'symbols', path: n.path });
      else if (act === 'open') open(n.path, 1);
      else if (act === 'reveal') open(n.path, 1);
    }));
  }

  function archEdgeDetails(e) {
    const s = ARCH_EDGES[e.kind] || { label: e.kind };
    const kinds = e.kinds ? Object.entries(e.kinds).map(([k, v]) => `${v} ${k}`).join(', ') : '';
    const view = archStack[archStack.length - 1];
    const name = (id) => (view.nodes.find((n) => n.id === id) || { label: id }).label;
    const heuristic = ['http', 'grpc', 'process', 'datastore', 'model API'].includes(e.kind)
      ? ' Detected from patterns in the code, so check the evidence.' : '';
    $('arch-details').innerHTML = `<h3>${escapeHtml(name(e.source))} → ${escapeHtml(name(e.target))}</h3>`
      + `<p class="hint">${escapeHtml(s.label)}${kinds ? ` (${escapeHtml(kinds)})` : ''}.${heuristic}</p>`
      + `<p><code>${escapeHtml(e.label || '')}</code></p>` + evidenceList(e.evidence);
  }

  $('arch-compose').addEventListener('change', () => { if (archStack.length) renderArchitecture(); });
  $('arch-infra').addEventListener('change', () => { if (archStack.length) renderArchitecture(); });
  $('arch-mermaid').addEventListener('click', () => {
    const system = archStack[0];
    if (system && system.mermaid) vscode.postMessage({ type: 'copy', text: system.mermaid });
  });

  function showComponents(components) {
    if (!components || !components.length) return;
    $('components-section').hidden = false;
    $('components').innerHTML = components.map((c) => `<li><code>${escapeHtml(c.name)}</code> <span class="why">relevance ${pct(c.score)}</span></li>`).join('');
  }

  function onResult(kind, r) {
    $('status').textContent = '';
    if (kind === 'architecture') {
      showArchitecture(r);
      return;
    }
    if (kind === 'ask') {
      $('answer-section').hidden = false;
      $('answer').innerHTML = markdown(r.answer || answerText);
    }
    if (kind === 'edit') {
      $('answer-section').hidden = false;
      const files = r.files.map((f) => `- \`${f.path}\`${f.created ? ' (new file)' : ''}`).join('\n');
      $('answer').innerHTML = markdown(`${r.summary || ''}\n\nFiles:\n${files || '- none'}`);
      if (r.skipped && r.skipped.length) note('warning', `Skipped: ${r.skipped.join('; ')}`);
      if (r.syntaxErrors && r.syntaxErrors.length) note('error', `Still doesn't parse: ${r.syntaxErrors.join('; ')}`);
    }
    if (kind === 'ask') {
      showComponents(r.components);
      showGraphAnswer(r.graphAnswer);
      showRelations(r.relations, r.folderLinks);
    }
    if (r.impact) showImpact(r.impact);
    showGraph(r.graph);
    showContext(r.context, kind === 'graph' ? `Nearest symbols (${(r.context || []).length})` : undefined);
  }

  document.addEventListener('click', (e) => {
    const a = e.target.closest('a.code-ref');
    if (!a) return;
    e.preventDefault();
    open(a.dataset.path, a.dataset.line);
  });

  window.addEventListener('message', (e) => {
    const m = e.data;
    if (m.type === 'reset') reset(m.kind, m.title);
    else if (m.type === 'event') onEvent(m.event);
    else if (m.type === 'result') onResult(m.kind, m.result);
    else if (m.type === 'note') note(m.level, m.text);
  });
  vscode.postMessage({ type: 'ready' });
}());
