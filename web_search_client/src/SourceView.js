import React, { useEffect, useState } from 'react';
import './App.css';
import { API_BASE } from './config';
import { formatSourceLabel, sourcePath } from './sourceLabel';

/**
 * Full text of one source, shown on its own page (index.js renders this instead of the app
 * when the URL has ?source=<uri>). Opened in a new tab from the Sources list. Uploaded PDF,
 * DOCX and MP3 files are stored as extracted text, so this is the text that was searched.
 */
function SourceView({ uri }) {
  const [state, setState] = useState({ loading: true, content: null, error: null });
  const name = formatSourceLabel(uri);

  useEffect(() => {
    document.title = `${name} · GraphRAG Search`;
    let active = true;
    fetch(`${API_BASE}/api/v1/chunk-source?uri=${encodeURIComponent(uri)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load this source (${res.status})`);
        return res.json();
      })
      .then((data) => {
        if (!active) return;
        if (data.found) setState({ loading: false, content: data.content, error: null });
        // Answers in saved chats keep their sources after the file is deleted
        else setState({ loading: false, content: null, error: 'This source is no longer available. Its file may have been deleted from your workspace.' });
      })
      .catch((err) => active && setState({ loading: false, content: null, error: err.message }));
    return () => { active = false; };
  }, [uri, name]);

  const words = state.content ? state.content.split(/\s+/).filter(Boolean).length : 0;

  return (
    <div className="app-container source-page">
      <header className="source-page-header">
        <p className="source-page-kicker">Source</p>
        <h1>{name}</h1>
        <p className="source-page-path" title={uri}>{sourcePath(uri)}</p>
        {state.content != null && (
          <p className="source-page-meta">{words.toLocaleString()} word{words === 1 ? '' : 's'}</p>
        )}
      </header>
      {state.loading && <p className="evidence-loading">Loading…</p>}
      {state.error && <p className="evidence-error" role="alert">{state.error}</p>}
      {state.content != null && <article className="source-page-text">{state.content}</article>}
    </div>
  );
}

export default SourceView;
