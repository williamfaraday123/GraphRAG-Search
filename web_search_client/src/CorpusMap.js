import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './CorpusMap.css';

// Categorical slots in fixed order (validated on the app's dark surface).
// Files keep their workspace position, so a file's colour never changes with the scope.
const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const OTHER = '#6b6b80';
const TYPE_COLORS = { name: SERIES[0], concept: SERIES[1] };
const TYPE_LABELS = { name: 'Name', concept: 'Concept' };

const VIEW_W = 640;
const VIEW_H = 400;
const PAD = 40;
// Every entity gets a label where it fits; the most mentioned are placed first and win
// any overlap (the rest show their label on hover or when a neighbour is selected)
const ALWAYS_LABELLED = 40;
const CHAR_W = 6.2; // approximate width of one 11px label character

const pos = (e) => ({ x: PAD + e.x * (VIEW_W - 2 * PAD), y: PAD + e.y * (VIEW_H - 2 * PAD) });
const radius = (e) => 5 + Math.sqrt(e.chunkCount) * 3;

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function CorpusMap({
  apiBase, corpusId, files, scopeFileIds, onClose,
  // What to highlight, from the results below the map: hovered, or pinned with "Show in map".
  //   { key, kind: 'chunk', chunkId }                    an evidence chunk's entities and facts
  //   { key, kind: 'path', facts: [{source, target, relation, ...}] }   a knowledge-graph reasoning path
  hoverHighlight = null, pinnedHighlight = null, onClearPinned, onBackToEvidence,
}) {
  const [state, setState] = useState({ loading: true, error: null, data: null });
  const [tab, setTab] = useState('entities');
  const [typeFilter, setTypeFilter] = useState('all');
  const [selectedId, setSelectedId] = useState(null);
  const [hover, setHover] = useState(null); // { id, x, y }
  const [openClusters, setOpenClusters] = useState(() => new Set());
  const [builtScope, setBuiltScope] = useState(scopeFileIds);
  const graphRef = useRef(null);

  const load = useCallback((ids) => {
    setState({ loading: true, error: null, data: null });
    setSelectedId(null);
    setBuiltScope(ids);
    const query = ids.length ? `?fileIds=${ids.map(encodeURIComponent).join(',')}` : '';
    fetch(`${apiBase}/api/v1/corpora/${encodeURIComponent(corpusId)}/map${query}`)
      .then(async (res) => {
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.message || `Could not build the corpus map (${res.status})`);
        return body;
      })
      .then((data) => setState({ loading: false, error: null, data }))
      .catch((err) => setState({ loading: false, error: err.message, data: null }));
  }, [apiBase, corpusId]);

  // Build once on open; afterwards the user rebuilds explicitly when the scope changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load(scopeFileIds); }, [load]);

  const scopeChanged = builtScope.join(',') !== scopeFileIds.join(',');
  const data = state.data;
  // "knowledge_graph": facts the local model extracted (directed, labelled edges).
  // "keywords": statistical fallback while extraction hasn't produced facts yet.
  const isKg = data?.mode === 'knowledge_graph';
  const progress = data?.progress;

  const fileColor = useMemo(() => {
    const map = {};
    files.forEach((f, i) => { map[f.fileId] = SERIES[i] || OTHER; });
    return map;
  }, [files]);

  const entityById = useMemo(() => {
    const map = {};
    (data?.entities || []).forEach((e) => { map[e.id] = e; });
    return map;
  }, [data]);

  const visibleEntities = useMemo(
    () => (data?.entities || []).filter((e) => typeFilter === 'all' || e.type === typeFilter),
    [data, typeFilter]
  );
  const visibleIds = useMemo(() => new Set(visibleEntities.map((e) => e.id)), [visibleEntities]);
  const visibleLinks = useMemo(
    () => (data?.relationships || []).filter((r) => visibleIds.has(r.source) && visibleIds.has(r.target)),
    [data, visibleIds]
  );

  const neighbours = useMemo(() => {
    const map = {};
    visibleLinks.forEach((r) => {
      (map[r.source] = map[r.source] || []).push({ id: r.target, rel: r });
      (map[r.target] = map[r.target] || []).push({ id: r.source, rel: r });
    });
    Object.values(map).forEach((list) => list.sort((a, b) => b.rel.weight - a.rel.weight));
    return map;
  }, [visibleLinks]);

  // A newly pinned highlight replaces the current selection so it shows
  const pinnedKey = pinnedHighlight?.key;
  useEffect(() => {
    if (!pinnedKey) return;
    setSelectedId(null);
    setTab('entities');
    setTypeFilter('all');
  }, [pinnedKey]);

  const active = hoverHighlight || pinnedHighlight;
  const showingPinned = Boolean(pinnedHighlight) && !hoverHighlight;
  const evidenceChunkId = active?.kind === 'chunk' ? active.chunkId : null;
  const evidence = useMemo(() => {
    if (!active || !data) return null;
    if (active.kind === 'path') {
      // Entities and lines on a reasoning path; relation per line (either direction)
      const keys = [...new Set(active.facts.flatMap((f) => [f.source, f.target]))];
      const relation = {};
      active.facts.forEach((f) => {
        relation[`${f.source}|${f.target}`] = f.relation;
        relation[`${f.target}|${f.source}`] = f.relation;
      });
      return {
        kind: 'path',
        facts: active.facts,
        ids: new Set(keys.filter((k) => visibleIds.has(k))),
        missing: keys.filter((k) => !entityById[k]).length,
        inMap: keys.some((k) => entityById[k]),
        relation,
      };
    }
    const ids = new Set(visibleEntities.filter((e) => e.chunkIds.includes(active.chunkId)).map((e) => e.id));
    const factCount = isKg
      ? visibleLinks.reduce((n, r) => n + r.examples.filter((ex) => ex.chunkId === active.chunkId).length, 0)
      : 0;
    const chunk = data.chunks?.[active.chunkId];
    return { kind: 'chunk', chunk, ids, factCount, inMap: Boolean(chunk) };
  }, [active, data, visibleEntities, visibleLinks, visibleIds, entityById, isKg]);

  const focusId = hover?.id || selectedId;
  const focusSet = useMemo(() => {
    if (!focusId) return null;
    return new Set([focusId, ...(neighbours[focusId] || []).map((n) => n.id)]);
  }, [focusId, neighbours]);
  // Hovering or selecting an entity in the map takes over from the evidence highlight
  const evidenceSet = !focusId && evidence?.inMap ? evidence.ids : null;
  const emphasis = focusSet || evidenceSet;

  // Labels are placed greedily, highest priority first (focused entity and its
  // neighbours, or the evidence chunk's entities, then the most mentioned),
  // skipping any that would overlap one already placed.
  const { shown: labelledIds, boxes: labelBoxes } = useMemo(() => {
    const order = [
      ...(focusId ? [focusId] : []),
      ...(emphasis ? visibleEntities.filter((e) => emphasis.has(e.id) && e.id !== focusId).map((e) => e.id) : []),
      ...visibleEntities.slice(0, ALWAYS_LABELLED).map((e) => e.id),
    ];
    const placed = [];
    const shown = new Set();
    order.forEach((id) => {
      if (shown.has(id)) return;
      const e = visibleEntities.find((v) => v.id === id);
      if (!e) return;
      const p = pos(e);
      const w = e.label.length * CHAR_W;
      const box = { x1: p.x - w / 2, x2: p.x + w / 2, y1: p.y - radius(e) - 17, y2: p.y - radius(e) - 3 };
      if (placed.some((b) => box.x1 < b.x2 && box.x2 > b.x1 && box.y1 < b.y2 && box.y2 > b.y1)) return;
      placed.push(box);
      shown.add(id);
    });
    return { shown, boxes: placed };
  }, [visibleEntities, focusId, emphasis]);

  // Line state: focused entity's lines, lines with a fact/sentence from the evidence chunk,
  // or the lines on a reasoning path
  const linkState = (r) => {
    if (focusId) return r.source === focusId || r.target === focusId ? 'active' : 'dim';
    if (evidenceSet) {
      const onIt = evidence.kind === 'path'
        ? Boolean(evidence.relation[`${r.source}|${r.target}`])
        : r.examples.some((ex) => ex.chunkId === evidenceChunkId);
      if (onIt) return 'active';
      return evidenceSet.has(r.source) && evidenceSet.has(r.target) ? '' : 'dim';
    }
    return '';
  };
  // Relation shown on a highlighted line: the path's fact, the evidence chunk's fact, or the strongest
  const lineRelation = (r) => {
    if (evidenceSet && !focusId) {
      if (evidence.kind === 'path') return evidence.relation[`${r.source}|${r.target}`];
      return r.examples.find((ex) => ex.chunkId === evidenceChunkId)?.relation;
    }
    return r.examples[0]?.relation;
  };

  // Relation labels on the highlighted lines (KG mode). An evidence chunk can light up many
  // facts at once, so each label is placed only where it overlaps no entity label or
  // relation label already placed; skipped ones are still listed when an entity is selected.
  const relationLabels = (() => {
    const out = {};
    if (!isKg) return out;
    const taken = [
      ...labelBoxes,
      ...visibleEntities.map((e) => {
        const p = pos(e);
        const r = radius(e) + 4;
        return { x1: p.x - r, x2: p.x + r, y1: p.y - r, y2: p.y + r };
      }),
    ];
    visibleLinks.forEach((r) => {
      if (linkState(r) !== 'active') return;
      const relation = lineRelation(r);
      if (!relation) return;
      const text = relation.length > 28 ? `${relation.slice(0, 27)}…` : relation;
      const a = pos(entityById[r.source]);
      const b = pos(entityById[r.target]);
      const x = (a.x + b.x) / 2;
      const y = (a.y + b.y) / 2 - 4;
      const w = text.length * CHAR_W * 0.9; // 10px italic
      const box = { x1: x - w / 2, x2: x + w / 2, y1: y - 10, y2: y + 3 };
      if (taken.some((t) => box.x1 < t.x2 && box.x2 > t.x1 && box.y1 < t.y2 && box.y2 > t.y1)) return;
      taken.push(box);
      out[`${r.source}|${r.target}`] = { x, y, text };
    });
    return out;
  })();
  const maxWeight = Math.max(1, ...visibleLinks.map((r) => r.weight));

  const selectEntity = (id) => {
    setTab('entities');
    if (id && entityById[id] && typeFilter !== 'all' && entityById[id].type !== typeFilter) setTypeFilter('all');
    setSelectedId((prev) => (prev === id ? null : id));
  };

  const showTooltip = (event, id) => {
    const box = graphRef.current?.getBoundingClientRect();
    if (!box) return;
    setHover({ id, x: event.clientX - box.left, y: event.clientY - box.top });
  };

  const fileNames = (ids) => ids.map((id) => files.find((f) => f.fileId === id)?.filename || 'Unknown file');

  const selected = selectedId ? entityById[selectedId] : null;
  const hovered = hover ? entityById[hover.id] : null;

  return (
    <section className="corpus-map" aria-label="Corpus map">
      <div className="corpus-map-header">
        <div>
          <h3>Corpus map</h3>
          {data && (
            <p className="corpus-map-stats">
              {plural(data.stats.files, 'file')} · {plural(data.stats.chunks, 'chunk')} ·{' '}
              {plural(data.stats.entities, 'entity').replace('entitys', 'entities')} ·{' '}
              {plural(data.stats.relationships, 'relationship')} · {plural(data.stats.clusters, 'cluster')}
            </p>
          )}
        </div>
        <div className="file-actions">
          <button type="button" className="file-action" onClick={() => load(scopeFileIds)} disabled={state.loading}>
            {state.loading ? 'Building…' : 'Rebuild'}
          </button>
          <button type="button" className="file-action" onClick={onClose}>Close</button>
        </div>
      </div>

      {scopeChanged && !state.loading && (
        <p className="corpus-map-note">File selection changed. Rebuild to map the {scopeFileIds.length
          ? plural(scopeFileIds.length, 'selected file') : 'whole workspace'}.</p>
      )}
      {state.loading && <p className="corpus-map-note">Reading chunks and finding entities and themes…</p>}
      {state.error && <p className="upload-error">{state.error}</p>}
      {data && progress?.enabled && progress.pending > 0 && (
        <p className="corpus-map-status" role="status">
          Extracting facts with the local model: {progress.done + progress.failed} of{' '}
          {plural(progress.total, 'chunk')} read.{' '}
          {isKg ? 'Rebuild later to add the rest.' : 'Until facts are ready, this is a quick keyword map. Rebuild later to see them.'}
        </p>
      )}
      {data && progress?.failed > 0 && (
        <p className="corpus-map-note">{plural(progress.failed, 'chunk')} could not be read by the model.</p>
      )}
      {data && !isKg && progress && !progress.enabled && (
        <p className="corpus-map-note">Fact extraction is turned off (KG_ENABLED=false): showing a keyword map.</p>
      )}

      {data && evidence && (
        <div className="corpus-map-evidence" role="status">
          <p>
            {evidence.kind === 'path' ? (
              evidence.inMap ? (
                <>
                  <strong>Reasoning path:</strong> {plural(evidence.facts.length, 'fact')} linking{' '}
                  {[...new Set(evidence.facts.flatMap((f) => [f.sourceLabel, f.targetLabel]))].join(', ')}.
                  {evidence.missing > 0 && ` ${plural(evidence.missing, 'entity').replace('entitys', 'entities')} not in this map (rebuild with its file selected).`}
                </>
              ) : (
                <>This reasoning path's entities aren't in this map. Rebuild with their files selected to see it.</>
              )
            ) : !evidence.chunk ? (
              <>The {showingPinned ? 'selected' : 'hovered'} evidence chunk isn't in this map.
                Rebuild with its file selected to see it.</>
            ) : (
              <>
                <strong>Evidence:</strong> {evidence.chunk.filename} · chunk {evidence.chunk.chunkIndex + 1}.{' '}
                {evidence.ids.size === 0
                  ? (typeFilter === 'all' ? 'No mapped entities in this chunk.' : 'No entities of this type in this chunk.')
                  : <>
                    {plural(evidence.ids.size, 'entity').replace('entitys', 'entities')} highlighted
                    {isKg ? `, ${plural(evidence.factCount, 'fact')} from this chunk` : ''}.
                  </>}
              </>
            )}
          </p>
          {showingPinned && (
            <div className="file-actions">
              {onBackToEvidence && (
                <button type="button" className="file-action" onClick={onBackToEvidence}>
                  {evidence.kind === 'path' ? 'Back to reasoning' : 'Back to evidence'}
                </button>
              )}
              {onClearPinned && (
                <button type="button" className="file-action" onClick={onClearPinned}>Clear</button>
              )}
            </div>
          )}
        </div>
      )}

      {data && (
        <>
          <div className="corpus-map-tabs" role="tablist" aria-label="Corpus map views">
            {[['entities', 'Entities & relationships'], ['clusters', 'Document clusters']].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                className={`corpus-map-tab ${tab === id ? 'active' : ''}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === 'entities' && (
            <div role="tabpanel" aria-label="Entities and relationships">
              <div className="corpus-map-controls">
                <div className="corpus-map-filter" role="group" aria-label="Entity type">
                  {[['all', 'All'], ['name', 'Names'], ['concept', 'Concepts']].map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      aria-pressed={typeFilter === id}
                      className={`corpus-map-chip ${typeFilter === id ? 'active' : ''}`}
                      onClick={() => { setTypeFilter(id); setSelectedId(null); }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <ul className="corpus-map-legend" aria-label="Legend">
                  {Object.entries(TYPE_LABELS).map(([type, label]) => (
                    <li key={type}><span className="swatch round" style={{ background: TYPE_COLORS[type] }} />{label}</li>
                  ))}
                  {isKg
                    ? <li><span className="swatch line arrow" />Fact stated in your files (arrow: subject → object)</li>
                    : <li><span className="swatch line" />Mentioned in the same sentence</li>}
                </ul>
              </div>

              {visibleEntities.length === 0 ? (
                <p className="corpus-map-note">No entities of this type were found.</p>
              ) : (
                <div className="corpus-map-graph" ref={graphRef} onMouseLeave={() => setHover(null)}>
                  <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img"
                    aria-label={`Graph of ${visibleEntities.length} entities and ${visibleLinks.length} relationships`}
                    onClick={(e) => { if (e.target === e.currentTarget) setSelectedId(null); }}>
                    <defs>
                      {['', '-active'].map((suffix) => (
                        <marker key={suffix} id={`cm-arrow${suffix}`} viewBox="0 0 10 10" refX="9" refY="5"
                          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                          <path d="M0,1 L9,5 L0,9 z" className={`arrow-head${suffix}`} />
                        </marker>
                      ))}
                    </defs>
                    <g className="links">
                      {visibleLinks.map((r) => {
                        const src = entityById[r.source];
                        const dst = entityById[r.target];
                        const a = pos(src);
                        const b = pos(dst);
                        const linkClass = linkState(r);
                        const active = linkClass === 'active';
                        // Facts are directed (subject -> object); an edge can hold facts both ways
                        const forward = isKg && r.examples.some((ex) => ex.from === r.source);
                        const backward = isKg && r.examples.some((ex) => ex.from === r.target);
                        // Stop lines at the node edge so arrowheads stay visible
                        const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
                        const ux = (b.x - a.x) / len;
                        const uy = (b.y - a.y) / len;
                        const ra = radius(src) + 3;
                        const rb = radius(dst) + 3;
                        const marker = `url(#cm-arrow${active ? '-active' : ''})`;
                        const label = relationLabels[`${r.source}|${r.target}`];
                        return (
                          <g key={`${r.source}|${r.target}`}>
                            <line
                              x1={a.x + ux * ra} y1={a.y + uy * ra} x2={b.x - ux * rb} y2={b.y - uy * rb}
                              className={`link ${linkClass}`}
                              strokeWidth={1 + (2 * r.weight) / maxWeight}
                              markerEnd={forward ? marker : undefined}
                              markerStart={backward ? marker : undefined}
                            />
                            {label && (
                              <text x={label.x} y={label.y} textAnchor="middle" className="link-label">{label.text}</text>
                            )}
                          </g>
                        );
                      })}
                    </g>
                    <g className="nodes">
                      {visibleEntities.map((e) => {
                        const p = pos(e);
                        const dim = emphasis && !emphasis.has(e.id);
                        const inEvidence = evidenceSet?.has(e.id);
                        const showLabel = labelledIds.has(e.id);
                        const r = radius(e);
                        return (
                          <g
                            key={e.id}
                            className={`node ${dim ? 'dim' : ''} ${selectedId === e.id ? 'selected' : ''} ${inEvidence ? 'in-evidence' : ''}`}
                            transform={`translate(${p.x},${p.y})`}
                            tabIndex={0}
                            role="button"
                            aria-label={`${e.label}, ${TYPE_LABELS[e.type]}, in ${plural(e.chunkCount, 'chunk')}`}
                            onMouseEnter={(ev) => showTooltip(ev, e.id)}
                            onMouseMove={(ev) => showTooltip(ev, e.id)}
                            onClick={() => selectEntity(e.id)}
                            onKeyDown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectEntity(e.id); } }}
                          >
                            <circle r={r + 8} className="hit" />
                            {inEvidence && <circle r={r + 4} className="evidence-ring" />}
                            <circle r={r} fill={TYPE_COLORS[e.type]} className="dot" />
                            {showLabel && (() => {
                              // Keep labels inside the plot near its left/right edges
                              const half = (e.label.length * CHAR_W) / 2;
                              const anchor = p.x - half < 2 ? 'start' : p.x + half > VIEW_W - 2 ? 'end' : 'middle';
                              const dx = anchor === 'start' ? -r : anchor === 'end' ? r : 0;
                              return <text x={dx} y={-r - 5} textAnchor={anchor} className="node-label">{e.label}</text>;
                            })()}
                          </g>
                        );
                      })}
                    </g>
                  </svg>
                  {hovered && (
                    <div
                      className="corpus-map-tooltip"
                      style={{ left: Math.min(hover.x + 14, (graphRef.current?.clientWidth || 600) - 220), top: hover.y + 14 }}
                    >
                      <strong>{hovered.label}</strong>
                      <span>{TYPE_LABELS[hovered.type]} · {plural(hovered.mentions, 'mention')} in {plural(hovered.chunkCount, 'chunk')}</span>
                      <span>{fileNames(hovered.fileIds).join(', ')}</span>
                      <span>{plural((neighbours[hovered.id] || []).length, 'linked entity').replace('entitys', 'entities')}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="corpus-map-entity-list" aria-label="All entities">
                {visibleEntities.map((e) => (
                  <button
                    key={e.id}
                    type="button"
                    className={`corpus-map-chip ${selectedId === e.id ? 'active' : ''}`}
                    aria-pressed={selectedId === e.id}
                    onClick={() => selectEntity(e.id)}
                  >
                    <span className="swatch round" style={{ background: TYPE_COLORS[e.type] }} />
                    {e.label} <span className="chip-count">{e.chunkCount}</span>
                  </button>
                ))}
              </div>

              {selected ? (
                <div className="corpus-map-detail" aria-live="polite">
                  <h4>{selected.label}</h4>
                  <p className="corpus-map-meta">
                    {TYPE_LABELS[selected.type]} · {plural(selected.mentions, 'mention')} in{' '}
                    {plural(selected.chunkCount, 'chunk')} · {fileNames(selected.fileIds).join(', ')}
                  </p>

                  <h5>{isKg ? 'Facts' : 'Relationships'}</h5>
                  {(neighbours[selected.id] || []).length === 0 ? (
                    <p className="corpus-map-note">
                      {isKg ? 'No facts link this to another mapped entity.'
                        : 'Not mentioned in the same sentence as another mapped entity.'}
                    </p>
                  ) : isKg ? (
                    <ul className="corpus-map-relations">
                      {(neighbours[selected.id] || []).flatMap(({ id, rel }) => rel.examples.map((ex, i) => {
                        // Write the fact as a sentence; the other entity is a link to it
                        const end = (key) => (key === selected.id
                          ? <strong>{selected.label}</strong>
                          : (
                            <button type="button" className="link-button" onClick={() => selectEntity(key)}>
                              {entityById[key]?.label || key}
                            </button>
                          ));
                        return (
                          <li key={`${id}-${i}`}>
                            <p className="corpus-map-fact">
                              {end(ex.from)} <span className="fact-relation">{ex.relation}</span> {end(ex.to)}
                            </p>
                            <blockquote>“{ex.sentence}” <cite>{ex.filename}</cite></blockquote>
                          </li>
                        );
                      }))}
                    </ul>
                  ) : (
                    <ul className="corpus-map-relations">
                      {(neighbours[selected.id] || []).map(({ id, rel }) => (
                        <li key={id}>
                          <button type="button" className="link-button" onClick={() => selectEntity(id)}>
                            {entityById[id].label}
                          </button>
                          <span className="corpus-map-meta"> · {plural(rel.weight, 'shared sentence')}
                            {rel.fileIds.length > 1 ? ` · across ${rel.fileIds.length} files` : ''}</span>
                          {rel.examples.map((ex, i) => (
                            <blockquote key={i}>
                              “{ex.sentence}” <cite>{ex.filename}</cite>
                            </blockquote>
                          ))}
                        </li>
                      ))}
                    </ul>
                  )}

                  <h5>Mentioned in</h5>
                  <ul className="corpus-map-chunks">
                    {selected.chunkIds.map((cid) => {
                      const c = data.chunks[cid];
                      return (
                        <li key={cid}>
                          {/* No file colour here: blue/orange already mean Name/Concept on this tab */}
                          <span className="corpus-map-meta">{c.filename} · chunk {c.chunkIndex + 1}</span>
                          <p>{c.snippet}</p>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : (
                <p className="corpus-map-note">Select an entity to see its relationships and the passages that mention it.</p>
              )}
            </div>
          )}

          {tab === 'clusters' && (
            <div role="tabpanel" aria-label="Document clusters">
              <p className="corpus-map-note">
                Chunks grouped by meaning. A theme drawn from several files shows where your documents cover the same ground.
              </p>
              {data.files.length > 1 && (
                <ul className="corpus-map-legend" aria-label="Files">
                  {data.files.map((f) => (
                    <li key={f.fileId}><span className="swatch" style={{ background: fileColor[f.fileId] || OTHER }} />{f.filename}</li>
                  ))}
                </ul>
              )}
              <ul className="corpus-map-clusters">
                {data.clusters.map((cl) => {
                  const total = cl.chunks.length;
                  const open = openClusters.has(cl.id);
                  return (
                    <li key={cl.id} className="corpus-map-cluster">
                      <div className="cluster-head">
                        <h4>Theme {cl.id}: {cl.label}</h4>
                        {cl.files.length > 1 && <span className="cluster-badge">Shared by {cl.files.length} files</span>}
                      </div>
                      <p className="corpus-map-meta">
                        {plural(total, 'chunk')} ·{' '}
                        {cl.files.map((f) => `${f.filename} (${f.chunkCount})`).join(', ')}
                      </p>
                      <div className="cluster-bar" role="img"
                        aria-label={cl.files.map((f) => `${f.filename}: ${f.chunkCount} of ${total} chunks`).join('; ')}>
                        {cl.files.map((f) => (
                          <span
                            key={f.fileId}
                            title={`${f.filename}: ${f.chunkCount} of ${total} chunks`}
                            style={{ flexGrow: f.chunkCount, background: fileColor[f.fileId] || OTHER }}
                          />
                        ))}
                      </div>
                      {cl.entityIds.length > 0 && (
                        <div className="cluster-entities">
                          {cl.entityIds.map((id) => entityById[id] && (
                            <button key={id} type="button" className="corpus-map-chip" onClick={() => selectEntity(id)}>
                              <span className="swatch round" style={{ background: TYPE_COLORS[entityById[id].type] }} />
                              {entityById[id].label}
                            </button>
                          ))}
                        </div>
                      )}
                      <button
                        type="button"
                        className="evidence-toggle"
                        aria-expanded={open}
                        onClick={() => setOpenClusters((prev) => {
                          const next = new Set(prev);
                          if (next.has(cl.id)) next.delete(cl.id); else next.add(cl.id);
                          return next;
                        })}
                      >
                        {open ? 'Hide chunks' : `Show ${plural(total, 'chunk')}`}
                      </button>
                      {open && (
                        <ul className="corpus-map-chunks">
                          {cl.chunks.map((c) => (
                            <li key={c.chunkId}>
                              <span className="corpus-map-meta">
                                <span className="swatch round" style={{ background: fileColor[c.fileId] || OTHER }} />
                                {c.filename} · chunk {c.chunkIndex + 1}
                              </span>
                              <p>{c.snippet}</p>
                            </li>
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default CorpusMap;
