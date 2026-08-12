import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// Empty string = relative URL, resolves against whatever host served this page.
// Only needed as an absolute URL for `npm start` dev server (set REACT_APP_API_URL
// in web_search_client/.env.local) since that runs on a different port than the API.
const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState(null);
  const [streamedText, setStreamedText] = useState('');
  const [sources, setSources] = useState([]);
  const [confidence, setConfidence] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);
  const eventSourceRef = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  const resetState = useCallback(() => {
    setStatus(null);
    setStreamedText('');
    setSources([]);
    setConfidence(null);
    setError(null);
    setDone(false);
  }, []);

  const handleSearch = useCallback((e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    resetState();
    setIsLoading(true);

    const controller = new AbortController();

    fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ prompt: query.trim(), topK: 5 }),
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
  }, [query, isLoading, resetState]);

  const handleSSEEvent = (eventType, data) => {
    switch (eventType) {
      case 'status':
        setStatus({ stage: data.stage, message: data.message });
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

  const stageIcons = {
    RETRIEVING: '\uD83D\uDD0D',    // 🔍
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

      <form className="search-form" onSubmit={handleSearch}>
        <div className="search-bar">
          <input
            type="text"
            className="search-input"
            placeholder="Ask anything — AI, science, history..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            autoFocus
          />
          <button
            type="submit"
            className="search-button"
            disabled={isLoading || !query.trim()}
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
          <p className="empty-hint">Type a question above to start searching</p>
        </div>
      )}
    </div>
  );
}

export default App;
