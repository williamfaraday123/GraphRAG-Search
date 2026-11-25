import { memo, useCallback, useState } from 'react';
import Markdown, { parseOptionChecks, renderInline, splitAnswer } from './Markdown';
import { formatSourceLabel, sourcePageHref, sourcePath, uploadFileId } from './sourceLabel';

const AGENT_LABELS = {
  rewriter: 'Rewriter',
  planner: 'Planner',
  retriever: 'Retriever',
  analyst: 'Critic',
  synthesizer: 'Synthesizer',
};

// The query processor's status stages, shown as numbered steps. A second retrieval round
// (after the critic finds a gap) goes back to step 2.
const STAGE_STEPS = ['Plan', 'Retrieve', 'Check', 'Write'];
const STAGE_INDEX = { PLANNING: 0, RETRIEVING: 1, ITERATING: 1, ANALYZING: 2, SYNTHESIZING: 3 };

/**
 * One question of the chat and everything found for it: progress while it runs, then the
 * answer, its sources, the knowledge-graph reasoning and the evidence chunks. Earlier turns
 * stay on the page unchanged, so their answers and sources can still be read and opened.
 *
 * turn: { id, question, searchedAs, status, agentSteps, text, sources, chunks, graphPaths,
 *         graphSeeds, confidence, error, done }
 * availableFileIds: ids of the workspace's files right now (null = not known yet, or the wiki).
 *   Sources and evidence from a file that has since been deleted keep their stored text but
 *   say the file is no longer available, instead of offering links that would fail.
 *
 * Memoised: a streamed token changes only the turn being written, so earlier turns (and other
 * chats) are not redrawn while an answer streams in.
 */
function ChatTurn({
  turn, mode, mapOpen, pinnedHighlight, hoverProps, toggleShowInMap, sourceCache, fetchChunkSource,
  availableFileIds,
}) {
  const [expandedChunks, setExpandedChunks] = useState(() => new Set());
  const {
    id, question, searchedAs, status, agentSteps, text, sources, chunks, graphPaths, graphSeeds,
    confidence, error, done,
  } = turn;
  const isGone = (fileId) => Boolean(availableFileIds && fileId && !availableFileIds.has(fileId));

  const toggleChunkSource = useCallback((chunk, idx) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
    if (chunk.source && !sourceCache[chunk.source]) fetchChunkSource(chunk.source);
  }, [sourceCache, fetchChunkSource]);

  // Highlight keys and anchors carry the turn id: the same chunk can be evidence in several turns
  const chunkHighlight = (chunkId) => ({
    key: `chunk:${id}:${chunkId}`, kind: 'chunk', chunkId, anchorId: `evidence-${id}-${chunkId}`,
  });
  // Shared facts (file-similarity answers) or a reasoning path; the map banner names which
  const pathLabel = graphPaths.length && graphPaths.every((p) => p.kind === 'shared') ? 'Shared facts' : 'Reasoning path';
  const pathHighlight = (index) => (index === 'all'
    ? { key: `path:${id}:all`, kind: 'path', label: pathLabel, facts: graphPaths.flatMap((p) => p.facts), anchorId: `graph-reasoning-${id}` }
    : { key: `path:${id}:${index}`, kind: 'path', label: pathLabel, facts: graphPaths[index].facts, anchorId: `graph-path-${id}-${index}` });

  const stageIndex = STAGE_INDEX[status?.stage] ?? -1;
  const trace = agentSteps.length > 0 && (
    <ul className="agent-trace-list">
      {agentSteps.map((step, i) => (
        <li key={i} className="agent-trace-item">
          <span className="agent-trace-name">{AGENT_LABELS[step.agent] || step.agent}</span>
          {/* The critic's detail is the model's own Markdown: show its bold, not its asterisks */}
          <span className="agent-trace-detail">{renderInline((step.detail || '').replace(/\s+/g, ' '), `trace-${id}-${i}`)}</span>
        </li>
      ))}
    </ul>
  );

  return (
    <article className="turn" aria-label={`Question: ${question}`}>
      <div className="turn-question">
        <p className="turn-question-text">{question}</p>
      </div>
      {searchedAs && <p className="turn-searched-as">Searched as “{searchedAs}”</p>}

      {status && (
        <div className="progress" role="status">
          {stageIndex >= 0 && (
            <ol className="progress-steps">
              {STAGE_STEPS.map((label, i) => (
                <li
                  key={label}
                  className={i < stageIndex ? 'is-done' : i === stageIndex ? 'is-current' : ''}
                  aria-current={i === stageIndex ? 'step' : undefined}
                >
                  <span className="progress-num">{i + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          )}
          <p className="progress-message">{status.message}</p>
        </div>
      )}

      {/* While searching the log is open; afterwards it folds away under the answer's turn */}
      {trace && (done || error ? (
        <details className="agent-trace">
          <summary className="agent-trace-label">Search log ({agentSteps.length} steps)</summary>
          {trace}
        </details>
      ) : (
        <div className="agent-trace">
          <h3 className="agent-trace-label">Search log</h3>
          {trace}
        </div>
      ))}

      {error && <div className="error-bar" role="alert">{error}</div>}

      {(text || done) && (
        <div className="results">
          <div className="results-main">
            <div className="answer-section">
              <h2 className="answer-label">Answer</h2>
              {(() => {
                // The model writes "### Checked evidence" then "### Answer" (it reasons before it
                // answers). Show the answer first. Multiple choice: a summary, then every option
                // with its verdict. Otherwise: the checks one click away.
                const { answer, checks } = splitAnswer(text);
                const options = done ? parseOptionChecks(checks) : null;
                if (options) {
                  const supported = options.filter((o) => o.supported).length;
                  const titleId = `option-breakdown-title-${id}`;
                  return (
                    <>
                      <div className="answer-text answer-summary">
                        <Markdown text={answer} />
                      </div>
                      <section className="option-breakdown" aria-labelledby={titleId}>
                        <h3 id={titleId} className="option-breakdown-title">
                          Options evaluated
                          <span className="option-breakdown-count">{supported} of {options.length} supported</span>
                        </h3>
                        <ol className="option-list">
                          {options.map((o) => (
                            <li key={o.marker} className={`option-item ${o.supported ? 'is-supported' : 'is-rejected'}`}>
                              <div className="option-head">
                                <span className="option-marker">{o.marker}</span>
                                <span className="option-label">{o.label}</span>
                                <span className="option-verdict">
                                  <span aria-hidden="true">{o.supported ? '✓' : '✗'}</span>
                                  {o.supported ? 'Supported' : 'Not supported'}
                                </span>
                              </div>
                              {o.reason && <p className="option-reason">{renderInline(o.reason, `r-${id}-${o.marker}`)}</p>}
                              {o.evidence && <p className="option-evidence">{renderInline(o.evidence, `e-${id}-${o.marker}`)}</p>}
                            </li>
                          ))}
                        </ol>
                      </section>
                    </>
                  );
                }
                return (
                  <>
                    <div className="answer-text">
                      <Markdown text={answer} />
                    </div>
                    {checks && (
                      <details className="answer-checks">
                        <summary>How the answer was checked</summary>
                        <Markdown text={checks} />
                      </details>
                    )}
                  </>
                );
              })()}
            </div>

            {done && confidence !== null && confidence !== undefined && (
              <p className="confidence">
                Confidence {(confidence * 100).toFixed(0)}%
              </p>
            )}

            {sources.length > 0 && (
              <div className="sources-section">
                <h3>Sources</h3>
                <ul className="sources-list">
                  {sources.map((src, i) => (isGone(uploadFileId(src)) ? (
                    <li key={i} className="source-item">
                      <span className="source-link is-gone" title="This file was deleted from your workspace">
                        <span className="source-name">{formatSourceLabel(src)}</span>
                        <span className="source-text">No longer available: the file was deleted</span>
                      </span>
                    </li>
                  ) : (
                    <li key={i} className="source-item">
                      {/* Opens the full text on its own page in a new tab */}
                      <a
                        className="source-link"
                        href={sourcePageHref(src)}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={`Open the full text of ${formatSourceLabel(src)} in a new tab`}
                      >
                        <span className="source-name">{formatSourceLabel(src)}</span>
                        <span className="source-text">{sourcePath(src)}</span>
                        <span className="visually-hidden"> (opens in a new tab)</span>
                      </a>
                    </li>
                  )))}
                </ul>
              </div>
            )}
          </div>

          {done && (graphPaths.length > 0 || chunks.length > 0) && (
            <aside className="results-side" aria-label="Evidence">
              {graphPaths.length > 0 && (() => {
                const sharedFacts = pathLabel === 'Shared facts';
                const titleId = `graph-reasoning-title-${id}`;
                const allKey = pathHighlight('all').key;
                return (
                  <div className="graph-reasoning" id={`graph-reasoning-${id}`} aria-labelledby={titleId}>
                    <div className="graph-reasoning-header">
                      <h3 className="evidence-label" id={titleId}>
                        {sharedFacts ? 'Shared facts' : 'Reasoning path'}
                      </h3>
                      {mode === 'files' && (
                        <button
                          type="button"
                          className="evidence-toggle"
                          aria-pressed={pinnedHighlight?.key === allKey}
                          onClick={() => toggleShowInMap(pathHighlight('all'))}
                        >
                          {pinnedHighlight?.key === allKey ? 'Stop highlighting in map' : 'Show all in map'}
                        </button>
                      )}
                    </div>
                    {sharedFacts ? (
                      <p className="graph-reasoning-note">
                        The same claim found in both files' knowledge graphs, matched by meaning even when worded
                        differently. Facts are extracted automatically, so hover over one to see its sentence.
                      </p>
                    ) : (
                      <p className="graph-reasoning-note">
                        Facts the knowledge graph followed{graphSeeds.length ? ` from ${graphSeeds.join(', ')}` : ''} to
                        find the evidence below. They were extracted automatically and given to the model with the
                        text, so check them against the evidence.
                      </p>
                    )}
                    <ol className="graph-paths">
                      {graphPaths.map((path, i) => {
                        const highlight = pathHighlight(i);
                        const pinned = pinnedHighlight?.key === highlight.key;
                        return (
                          <li
                            key={i}
                            id={highlight.anchorId}
                            className={`graph-path ${pinned ? 'in-map' : ''}`}
                            {...(mode === 'files' ? hoverProps(highlight) : {})}
                          >
                            <div className="graph-path-facts">
                              {path.facts.map((f, j) => (
                                <p key={j} className="graph-fact">
                                  {j > 0 && (
                                    <span className="graph-fact-then">{path.kind === 'shared' ? 'same as' : 'then'}</span>
                                  )}
                                  <strong>{f.sourceLabel}</strong> <span className="fact-relation">{f.relation}</span>{' '}
                                  <strong>{f.targetLabel}</strong>
                                  <span className="graph-fact-file" title={f.evidence}> · {f.filename}</span>
                                </p>
                              ))}
                            </div>
                            {mode === 'files' && (
                              <button
                                type="button"
                                className="evidence-toggle"
                                aria-pressed={pinned}
                                onClick={() => toggleShowInMap(highlight)}
                              >
                                {pinned ? 'Stop highlighting' : 'Show in map'}
                              </button>
                            )}
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                );
              })()}

              {chunks.length > 0 && (
                <div className="evidence-section">
                  <h3 className="evidence-label">Evidence ({chunks.length} chunks)</h3>
                  {mapOpen && (
                    <p className="evidence-map-hint">
                      Hover over a chunk to highlight its entities in the corpus map, or use Show in map.
                    </p>
                  )}
                  <ul className="evidence-list">
                    {chunks.map((c, i) => {
                      const cached = c.source ? sourceCache[c.source] : null;
                      const isExpanded = expandedChunks.has(i);
                      const gone = isGone(uploadFileId(c.source) || (c.chunkId || '').split(':')[0]);
                      const mappable = mode === 'files' && Boolean(c.chunkId) && !gone;
                      const highlight = c.chunkId ? chunkHighlight(c.chunkId) : null;
                      const pinned = mappable && pinnedHighlight?.key === highlight.key;
                      return (
                        <li
                          key={c.chunkId || i}
                          id={highlight ? highlight.anchorId : undefined}
                          className={`evidence-item ${pinned ? 'in-map' : ''}`}
                          {...(mappable ? hoverProps(highlight) : {})}
                        >
                          <div className="evidence-header">
                            <span className="evidence-source">{formatSourceLabel(c.source)}</span>
                            {c.foundByGraph && (
                              <span className="evidence-badge-graph"
                                title="Vector search did not return this chunk; a knowledge-graph fact path led to it">
                                Found via knowledge graph
                              </span>
                            )}
                            {gone && (
                              <span className="evidence-badge-gone" title="The text below is what was found when the question was asked">
                                File deleted
                              </span>
                            )}
                          </div>
                          <p className="evidence-snippet">{c.content}</p>
                          <div className="evidence-scores">
                            <span>Semantic {Math.round((c.milvusScore || 0) * 100)}%</span>
                            <span>Authority {Math.round((c.pagerankScore || 0) * 100)}%</span>
                            {c.graphScore > 0 && (
                              <span>Graph {Math.round(c.graphScore * 100)}%</span>
                            )}
                            <span className="score-hybrid">Hybrid {Math.round((c.hybridScore || 0) * 100)}%</span>
                          </div>
                          <div className="evidence-actions">
                            {c.source && !gone && (
                              <button
                                type="button"
                                className="evidence-toggle"
                                aria-expanded={isExpanded}
                                onClick={() => toggleChunkSource(c, i)}
                              >
                                {isExpanded ? 'Hide full source' : 'View full source'}
                              </button>
                            )}
                            {mappable && (
                              <button
                                type="button"
                                className="evidence-toggle"
                                aria-pressed={pinned}
                                onClick={() => toggleShowInMap(highlight)}
                              >
                                {pinned ? 'Stop highlighting in map' : 'Show in map'}
                              </button>
                            )}
                          </div>
                          {isExpanded && (
                            <div className="evidence-full-source">
                              {cached?.loading && <span className="evidence-loading">Loading…</span>}
                              {cached?.error && <span className="evidence-error">{cached.error}</span>}
                              {cached?.content && <pre className="evidence-full-text">{cached.content}</pre>}
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </aside>
          )}
        </div>
      )}
    </article>
  );
}

export default memo(ChatTurn);
