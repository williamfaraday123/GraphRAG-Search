import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// Empty string = relative URL, resolves against whatever host served this page.
// Only needed as an absolute URL for `npm start` dev server (set REACT_APP_API_URL
// in web_search_client/.env.development.local) since that runs on a different port than
// the API. Don't use .env.local: `npm run build` reads it too and would bake the URL in.
const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('wiki');
  const [corpusId, setCorpusId] = useState(
    () => window.localStorage.getItem('graphrag-corpus-id') || ''
  );
  const [corpusError, setCorpusError] = useState(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState(null);
  const [status, setStatus] = useState(null);
  const [agentSteps, setAgentSteps] = useState([]);
  const [streamedText, setStreamedText] = useState('');
  const [sources, setSources] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [expandedChunks, setExpandedChunks] = useState(new Set());
  const [sourceCache, setSourceCache] = useState({}); // uri -> { loading, content, error }
  const [confidence, setConfidence] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);
  const eventSourceRef = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    if (!corpusId) return undefined;
    let active = true;
    const loadFiles = async () => {
      try {
        const response = await fetch(
          `${API_BASE}/api/v1/corpora/${encodeURIComponent(corpusId)}/files`
        );
        if (!response.ok) throw new Error(`Could not load uploaded files (${response.status})`);
        const files = await response.json();
        if (active) {
          setUploadedFiles(files);
          setCorpusError(null);
        }
      } catch (err) {
        if (active) setCorpusError(err.message);
      }
    };

    loadFiles();
    return () => { active = false; };
  }, [corpusId]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  const resetState = useCallback(() => {
    setStatus(null);
    setAgentSteps([]);
    setStreamedText('');
    setSources([]);
    setChunks([]);
    setExpandedChunks(new Set());
    setSourceCache({});
    setConfidence(null);
    setError(null);
    setDone(false);
  }, []);

  const chooseMode = useCallback(async (nextMode) => {
    setMode(nextMode);
    resetState();
    if (nextMode !== 'files' || corpusId) return;

    setCreatingWorkspace(true);
    setCorpusError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/corpora`, { method: 'POST' });
      if (!response.ok) throw new Error(`Could not create workspace (${response.status})`);
      const workspace = await response.json();
      window.localStorage.setItem('graphrag-corpus-id', workspace.corpusId);
      setCorpusId(workspace.corpusId);
    } catch (err) {
      setCorpusError(err.message);
    } finally {
      setCreatingWorkspace(false);
    }
  }, [corpusId, resetState]);

  const handleUpload = useCallback(async (e) => {
    e.preventDefault();
    const selectedFiles = Array.from(e.target.elements.files.files || []);
    if (!selectedFiles.length || uploading) return;
    if (!corpusId) {
      setUploadFeedback({ error: 'Your upload workspace is still being prepared.' });
      return;
    }

    setUploading(true);
    setUploadFeedback(null);
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append('files', file));
      const response = await fetch(
        `${API_BASE}/api/v1/corpora/${encodeURIComponent(corpusId)}/files`,
        { method: 'POST', body: formData }
      );
      if (!response.ok) throw new Error(`Upload failed (${response.status})`);
      const result = await response.json();
      const successful = (result.files || []).filter((file) => file.success);
      const failed = (result.files || []).filter((file) => !file.success);
      if (successful.length) {
        setUploadedFiles((previous) => [...previous, ...successful]);
      }
      setUploadFeedback({
        message: successful.length
          ? `${successful.length} file${successful.length === 1 ? '' : 's'} indexed and ready.`
          : null,
        error: failed.map((file) => `${file.filename}: ${file.message}`).join(' '),
      });
      e.target.reset();
    } catch (err) {
      setUploadFeedback({ error: err.message });
    } finally {
      setUploading(false);
    }
  }, [corpusId, uploading]);

  const handleSearch = useCallback((e) => {
    e.preventDefault();
    if (!query.trim() || isLoading || (mode === 'files' && !uploadedFiles.length)) return;

    resetState();
    setIsLoading(true);

    const controller = new AbortController();

    fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        prompt: query.trim(),
        topK: 5,
        corpusId: mode === 'wiki' ? 'wiki' : corpusId,
      }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done: readerDone, value } = await reader.read();
          if (readerDone) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let eventType = '';
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              try {
                const data = JSON.parse(line.slice(5).trim());
                handleSSEEvent(eventType, data);
              } catch (err) { /* skip malformed */ }
            }
          }
        }
      })
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      })
      .finally(() => setIsLoading(false));

    eventSourceRef.current = { close: () => controller.abort() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, isLoading, mode, corpusId, uploadedFiles.length, resetState]);

  const handleSSEEvent = (eventType, data) => {
    switch (eventType) {
      case 'status':
        setStatus({ stage: data.stage, message: data.message });
        break;
      case 'agent_step':
        setAgentSteps((prev) => [...prev, { agent: data.agent, detail: data.detail }]);
        break;
      case 'chunk':
        setStreamedText((prev) => prev + data.content);
        if (resultRef.current) resultRef.current.scrollTop = resultRef.current.scrollHeight;
        break;
      case 'done':
        setDone(true);
        setStatus(null);
        setIsLoading(false);
        setSources(data.sources || []);
        setChunks(data.chunks || []);
        setConfidence(data.confidence);
        break;
      case 'error':
        setError(data.message);
        setIsLoading(false);
        break;
      default:
        break;
    }
  };

  const fetchChunkSource = useCallback((uri) => {
    setSourceCache((prev) => ({ ...prev, [uri]: { loading: true, content: null, error: null } }));
    fetch(`${API_BASE}/api/v1/chunk-source?uri=${encodeURIComponent(uri)}`)
      .then((res) => res.json())
      .then((data) => {
        setSourceCache((prev) => ({
          ...prev,
          [uri]: data.found
            ? { loading: false, content: data.content, error: null }
            : { loading: false, content: null, error: data.error || 'Source not found' },
        }));
      })
      .catch((err) => {
        setSourceCache((prev) => ({ ...prev, [uri]: { loading: false, content: null, error: err.message } }));
      });
  }, []);

  const toggleChunkSource = useCallback((chunk, idx) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
    if (chunk.source && !sourceCache[chunk.source]) {
      fetchChunkSource(chunk.source);
    }
  }, [sourceCache, fetchChunkSource]);

  const formatSourceLabel = (source) => {
    if (!source) return 'Unknown source';
    return source.startsWith('s3://') ? source.slice('s3://'.length) : source;
  };

  const topAuthorityIndex = chunks.length > 0
    ? chunks.reduce((best, c, i, arr) => (c.pagerankScore > arr[best].pagerankScore ? i : best), 0)
    : -1;

  const agentMeta = {
    planner: { icon: '\uD83E\uDDED', label: 'Planner' },
    retriever: { icon: '\uD83D\uDD0D', label: 'Retriever' },
    analyst: { icon: '\uD83E\uDDD0', label: 'Critic' },
    synthesizer: { icon: '\u270D\uFE0F', label: 'Synthesizer' },
  };

  const stageIcons = {
    RETRIEVING: '\uD83D\uDD0D',    //🔍
    ITERATING: '\uD83D\uDD04',     // 🔄
    SYNTHESIZING: '\u2728',        // ✨
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon">🧠</span>
          <h1>GraphRAG Search</h1>
        </div>
        <p className="brand-subtitle">
          Hybrid Vector + Knowledge Graph Retrieval with Pregel Authority Ranking
        </p>
      </header>

      <div className="mode-selector" role="group" aria-label="Search mode">
        <button
          type="button"
          className={`mode-button ${mode === 'wiki' ? 'active' : ''}`}
          aria-pressed={mode === 'wiki'}
          onClick={() => chooseMode('wiki')}
        >
          Wiki dump
        </button>
        <button
          type="button"
          className={`mode-button ${mode === 'files' ? 'active' : ''}`}
          aria-pressed={mode === 'files'}
          onClick={() => chooseMode('files')}
        >
          My files
        </button>
      </div>

      {mode === 'files' && (
        <section className="upload-panel" aria-label="Upload files">
          <div className="upload-heading">
            <div>
              <h2>Ask questions about your files</h2>
              <p>Answers in this mode use your uploaded files only.</p>
            </div>
            <span className="upload-format">TXT · MD · max 5 MB each</span>
          </div>
          <form className="upload-form" onSubmit={handleUpload}>
            <input
              type="file"
              name="files"
              accept=".txt,.md,text/plain,text/markdown"
              multiple
              disabled={uploading || creatingWorkspace || !corpusId}
              aria-label="Choose text or Markdown files"
            />
            <button type="submit" className="upload-button" disabled={uploading || creatingWorkspace || !corpusId}>
              {creatingWorkspace ? 'Preparing workspace…' : uploading ? 'Indexing…' : 'Upload and index'}
            </button>
          </form>
          {corpusError && <p className="upload-error">{corpusError}</p>}
          {uploadFeedback?.message && <p className="upload-success">{uploadFeedback.message}</p>}
          {uploadFeedback?.error && <p className="upload-error">{uploadFeedback.error}</p>}
          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              <h3>Indexed files ({uploadedFiles.length})</h3>
              <ul>
                {uploadedFiles.map((file) => (
                  <li key={file.fileId || file.filename}>
                    <span>{file.filename}</span>
                    <span>{file.chunkCount ? `${file.chunkCount} chunks · ` : ''}Ready</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      <form className="search-form" onSubmit={handleSearch}>
        <div className="search-bar">
          <input
            type="text"
            className="search-input"
            placeholder={mode === 'wiki'
              ? 'Ask about the wiki dump…'
              : 'Ask a question about your uploaded files…'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            autoFocus
          />
          <button
            type="submit"
            className="search-button"
            disabled={isLoading || !query.trim() || (mode === 'files' && !uploadedFiles.length)}
          >
            {isLoading ? (
              <span className="spinner" />
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
            )}
          </button>
        </div>
      </form>

      {status && (
        <div className={`status-bar stage-${(status.stage || '').toLowerCase()}`}>
          <span className="status-icon">{stageIcons[status.stage] || '⏳'}</span>
          <span className="status-text">
            <strong>{status.stage}</strong> — {status.message}
          </span>
        </div>
      )}

      {agentSteps.length > 0 && (
        <div className="agent-trace">
          <h3 className="agent-trace-label">Agent Trace</h3>
          <ul className="agent-trace-list">
            {agentSteps.map((step, i) => {
              const meta = agentMeta[step.agent] || { icon: '🤖', label: step.agent };
              return (
                <li key={i} className="agent-trace-item">
                  <span className="agent-trace-icon">{meta.icon}</span>
                  <span className="agent-trace-name">{meta.label}</span>
                  <span className="agent-trace-detail">{step.detail}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {error && (
        <div className="error-bar">
          <span>⚠️</span> {error}
        </div>
      )}

      {(streamedText || done) && (
        <div className="results-panel" ref={resultRef}>
          <div className="answer-section">
            <h2 className="answer-label">Answer</h2>
            <div className="answer-text">
              {streamedText}
              {isLoading && <span className="cursor-blink">▌</span>}
            </div>
          </div>

          {done && confidence !== null && (
            <div className="metadata-row">
              <span className="confidence-badge">
                Confidence: {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}

          {done && chunks.length > 0 && (
            <div className="evidence-section">
              <h3 className="evidence-label">Evidence ({chunks.length} chunks)</h3>
              <ul className="evidence-list">
                {chunks.map((c, i) => {
                  const cached = c.source ? sourceCache[c.source] : null;
                  const isExpanded = expandedChunks.has(i);
                  return (
                    <li key={c.chunkId || i} className="evidence-item">
                      <div className="evidence-header">
                        {i === topAuthorityIndex && (
                          <span className="evidence-badge">🏆 Top Authority</span>
                        )}
                        <span className="evidence-source">{formatSourceLabel(c.source)}</span>
                      </div>
                      <p className="evidence-snippet">{c.content}</p>
                      <div className="evidence-scores">
                        <span className="score-pill score-semantic">Semantic {Math.round((c.milvusScore || 0) * 100)}%</span>
                        <span className="score-pill score-authority">Authority {Math.round((c.pagerankScore || 0) * 100)}%</span>
                        <span className="score-pill score-hybrid">Hybrid {Math.round((c.hybridScore || 0) * 100)}%</span>
                      </div>
                      {c.source && (
                        <button
                          type="button"
                          className="evidence-toggle"
                          onClick={() => toggleChunkSource(c, i)}
                        >
                          {isExpanded ? 'Hide full source ▲' : 'View full source ▼'}
                        </button>
                      )}
                      {isExpanded && (
                        <div className="evidence-full-source">
                          {cached?.loading && <span className="evidence-loading">Loading…</span>}
                          {cached?.error && <span className="evidence-error">⚠️ {cached.error}</span>}
                          {cached?.content && <pre className="evidence-full-text">{cached.content}</pre>}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {sources.length > 0 && (
            <div className="sources-section">
              <h3>Sources</h3>
              <ul className="sources-list">
                {sources.map((src, i) => (
                  <li key={i} className="source-item">
                    <span className="source-icon">📄</span>
                    <span className="source-text">{src}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {!streamedText && !isLoading && !error && (
        <div className="empty-state">
          <div className="empty-icon">⚡</div>
          <p>Powered by Milvus Vector DB, Neo4j Knowledge Graph &amp; LangGraph Pregel Engine</p>
          <p className="empty-hint">
            {mode === 'files' && !uploadedFiles.length
              ? 'Upload one or more text or Markdown files to get started'
              : 'Type a question above to start searching'}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
