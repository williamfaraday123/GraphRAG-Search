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
    ['answer-section', 'graph-answer-section', 'relations-section', 'impact-section', 'graph-section', 'context-section']
      .forEach((id) => { $(id).hidden = true; });
    ['graph-answer', 'folder-links', 'relations'].forEach((id) => { $(id).innerHTML = ''; });
    if (cy) { cy.destroy(); cy = null; }
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
    symbolList($('impact'), impact, (a) => `${a.direct ? 'direct dependent' : 'depends on it indirectly'} · PPR ${a.score.toFixed(3)}`);
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

  function onResult(kind, r) {
    $('status').textContent = '';
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
