import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';
import ChatTurn from './ChatTurn';
import CorpusMap from './CorpusMap';
import { API_BASE } from './config';

// Follow-ups: the last few questions and the start of their answers go with each search, so
// the query processor can rewrite "When did they buy it?" (it uses at most 3 turns)
const MAX_TURNS = 3;
const MAX_ANSWER_CHARS = 300;

function trimAnswer(text) {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (flat.length <= MAX_ANSWER_CHARS) return flat;
  const cut = flat.slice(0, MAX_ANSWER_CHARS);
  const space = cut.lastIndexOf(' ');
  return `${space > MAX_ANSWER_CHARS / 2 ? cut.slice(0, space) : cut}…`;
}

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('files');
  const [corpusId, setCorpusId] = useState(
    () => window.localStorage.getItem('graphrag-corpus-id') || ''
  );
  const [corpusError, setCorpusError] = useState(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState(null);
  // Checked files: the targets for Preview/Delete, and the query scope (none checked = all files)
  const [selectedFileIds, setSelectedFileIds] = useState(() => new Set());
  const [previewIds, setPreviewIds] = useState([]);
  const [previewCache, setPreviewCache] = useState({}); // fileId -> { loading, content, error }
  const [deletingFiles, setDeletingFiles] = useState(false);
  const [fileActionError, setFileActionError] = useState(null);
  const selectAllRef = useRef(null);
  const [showMap, setShowMap] = useState(false);
  // What the corpus map highlights (an evidence chunk or a reasoning path): while hovered,
  // or pinned by "Show in map". { key, kind, anchorId, chunkId | facts }
  const [hoverHighlight, setHoverHighlight] = useState(null);
  const [pinnedHighlight, setPinnedHighlight] = useState(null);
  const [scrollToMap, setScrollToMap] = useState(0);
  const [mapRequest, setMapRequest] = useState(null); // { id, fileIds }: one per Map click
  const mapRef = useRef(null);
  const [scrollToPreview, setScrollToPreview] = useState(0);
  const previewRef = useRef(null);
  const [sourceCache, setSourceCache] = useState({}); // uri -> { loading, content, error }
  const [isLoading, setIsLoading] = useState(false);
  // The chat on screen, oldest first: each question with its progress, answer and evidence
  // (ChatTurn), or { id, divider: true } where "New topic" was pressed
  const [turns, setTurns] = useState([]);
  // Earlier turns sent with each follow-up: {question, answer (trimmed), entityKeys}, oldest first
  const [conversation, setConversation] = useState([]);
  const turnCounter = useRef(0);
  const eventSourceRef = useRef(null);
  const scrollRef = useRef(null);       // the scrolling chat area
  const latestTurnRef = useRef(null);
  const [atBottom, setAtBottom] = useState(true);
  const inputRef = useRef(null);

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
          // Drop selections for files that no longer exist (e.g. deleted in another tab)
          const ids = new Set(files.map((f) => f.fileId));
          setSelectedFileIds((prev) => new Set([...prev].filter((id) => ids.has(id))));
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

  // Clears the whole chat (switching between My files and the wiki)
  const clearChat = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    setTurns([]);
    setConversation([]);
    setSourceCache({});
    setHoverHighlight(null);
    setPinnedHighlight(null);
    setIsLoading(false);
  }, []);

  const updateTurn = useCallback((id, change) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...(typeof change === 'function' ? change(t) : change) } : t)));
  }, []);

  const mapOpen = showMap && mode === 'files';

  // A new question scrolls up to the top of the view, so its answer is written below it
  useEffect(() => {
    if (latestTurnRef.current) latestTurnRef.current.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }, [turns.length]);

  const onChatScroll = useCallback(() => {
    const el = scrollRef.current;
    if (el) setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }, []);
  // The answer grows as it streams in: re-check whether the end is still in view
  useEffect(onChatScroll, [turns, onChatScroll]);

  const scrollToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo?.({ top: el.scrollHeight, behavior: 'smooth' });
  }, []);

  // After "Show in map" the map may only just be mounting, so scroll once it exists
  useEffect(() => {
    if (scrollToMap && mapRef.current) {
      mapRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [scrollToMap]);

  // Same for the file preview opened by the Preview button
  useEffect(() => {
    if (scrollToPreview && previewRef.current) {
      previewRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [scrollToPreview]);

  // Like Preview: every Map click builds the map for the files checked right now
  // (none checked = whole workspace), picking up facts extracted since the last build.
  const requestMap = useCallback(() => {
    const fileIds = uploadedFiles.filter((f) => selectedFileIds.has(f.fileId)).map((f) => f.fileId);
    setMapRequest((prev) => ({ id: (prev?.id || 0) + 1, fileIds }));
    setShowMap(true);
  }, [uploadedFiles, selectedFileIds]);

  const openMap = useCallback(() => {
    requestMap();
    setScrollToMap((n) => n + 1);
  }, [requestMap]);

  const toggleShowInMap = useCallback((highlight) => {
    if (pinnedHighlight?.key === highlight.key) {
      setPinnedHighlight(null);
      return;
    }
    setPinnedHighlight(highlight);
    setHoverHighlight(null);
    if (!showMap) requestMap(); // an open map keeps its current build
    setScrollToMap((n) => n + 1);
  }, [pinnedHighlight, showMap, requestMap]);

  const backToEvidence = useCallback(() => {
    const item = pinnedHighlight && document.getElementById(pinnedHighlight.anchorId);
    if (item) item.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [pinnedHighlight]);

  const closeMap = useCallback(() => {
    setShowMap(false);
    setPinnedHighlight(null);
    setHoverHighlight(null);
  }, []);

  // Hover handlers that highlight something in the open corpus map
  const hoverProps = (highlight) => (mapOpen
    ? {
      onMouseEnter: () => setHoverHighlight(highlight),
      onMouseLeave: () => setHoverHighlight(null),
      onFocus: () => setHoverHighlight(highlight),
      onBlur: () => setHoverHighlight(null),
    }
    : {});

  // "My files" needs an upload workspace: created on first use (on page load, since it's
  // the default mode) and remembered in local storage. The ref stops a second request while
  // one is in flight (React runs effects twice in development).
  const workspaceRequested = useRef(false);
  const ensureWorkspace = useCallback(async () => {
    if (corpusId || workspaceRequested.current) return;
    workspaceRequested.current = true;
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
      workspaceRequested.current = false; // switching back to My files retries
    } finally {
      setCreatingWorkspace(false);
    }
  }, [corpusId]);

  useEffect(() => {
    if (mode === 'files') ensureWorkspace();
  }, [mode, ensureWorkspace]);

  const chooseMode = useCallback((nextMode) => {
    if (nextMode !== mode) clearChat(); // the chat was about the other corpus
    setMode(nextMode);
  }, [mode, clearChat]);

  const handleUpload = useCallback(async (e) => {
    e.preventDefault();
    const selectedFiles = Array.from(e.target.elements.files.files || []);
    if (!selectedFiles.length || uploading) return;
    if (!corpusId) {
      setUploadFeedback({ error: 'Your upload workspace is still being prepared.' });
      return;
    }

    setUploading(true);
    const hasAudio = selectedFiles.some((file) => /\.mp3$/i.test(file.name));
    setUploadFeedback(hasAudio
      ? { message: 'Transcribing audio before indexing. This can take a few minutes…' }
      : null);
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

  const selectedCount = uploadedFiles.filter((f) => selectedFileIds.has(f.fileId)).length;
  const allSelected = uploadedFiles.length > 0 && selectedCount === uploadedFiles.length;

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = selectedCount > 0 && !allSelected;
    }
  }, [selectedCount, allSelected]);

  const toggleFileSelected = useCallback((fileId) => {
    setSelectedFileIds((prev) => {
      const next = new Set(prev);
      if (next.has(fileId)) next.delete(fileId); else next.add(fileId);
      return next;
    });
  }, []);

  const toggleAllFiles = useCallback(() => {
    setSelectedFileIds(allSelected ? new Set() : new Set(uploadedFiles.map((f) => f.fileId)));
  }, [allSelected, uploadedFiles]);

  const previewSelected = useCallback(() => {
    const ids = uploadedFiles.map((f) => f.fileId).filter((id) => selectedFileIds.has(id));
    setPreviewIds(ids);
    setScrollToPreview((n) => n + 1);
    ids.filter((id) => !previewCache[id]).forEach((id) => {
      setPreviewCache((prev) => ({ ...prev, [id]: { loading: true, content: null, error: null } }));
      fetch(`${API_BASE}/api/v1/corpora/${encodeURIComponent(corpusId)}/files/${encodeURIComponent(id)}`)
        .then(async (res) => {
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(data.message || `Could not load file (${res.status})`);
          setPreviewCache((prev) => ({ ...prev, [id]: { loading: false, content: data.content, error: null } }));
        })
        .catch((err) => {
          setPreviewCache((prev) => ({ ...prev, [id]: { loading: false, content: null, error: err.message } }));
        });
    });
  }, [uploadedFiles, selectedFileIds, previewCache, corpusId]);

  const deleteSelected = useCallback(async () => {
    const targets = uploadedFiles.filter((f) => selectedFileIds.has(f.fileId));
    if (!targets.length || deletingFiles) return;
    const names = targets.map((f) => f.filename).join(', ');
    if (!window.confirm(`Delete ${targets.length} file${targets.length === 1 ? '' : 's'} (${names})? `
      + 'They will be removed from search permanently.')) return;

    setDeletingFiles(true);
    setFileActionError(null);
    const deleted = new Set();
    const failures = [];
    for (const file of targets) {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/corpora/${encodeURIComponent(corpusId)}/files/${encodeURIComponent(file.fileId)}`,
          { method: 'DELETE' }
        );
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.message || `status ${res.status}`);
        }
        deleted.add(file.fileId);
      } catch (err) {
        failures.push(`${file.filename}: ${err.message}`);
      }
    }
    setUploadedFiles((prev) => prev.filter((f) => !deleted.has(f.fileId)));
    setSelectedFileIds((prev) => new Set([...prev].filter((id) => !deleted.has(id))));
    setPreviewIds((prev) => prev.filter((id) => !deleted.has(id)));
    setUploadFeedback(deleted.size
      ? { message: `Deleted ${deleted.size} file${deleted.size === 1 ? '' : 's'}.` }
      : null);
    if (failures.length) setFileActionError(`Could not delete ${failures.join('; ')}`);
    setDeletingFiles(false);
  }, [uploadedFiles, selectedFileIds, deletingFiles, corpusId]);

  const handleSearch = useCallback((e) => {
    e.preventDefault();
    if (!query.trim() || isLoading || (mode === 'files' && !uploadedFiles.length)) return;

    const asked = query.trim();
    turnCounter.current += 1;
    const turnId = turnCounter.current;
    setTurns((prev) => [...prev, {
      id: turnId, question: asked, searchedAs: '', status: null, agentSteps: [], text: '',
      sources: [], chunks: [], graphPaths: [], graphSeeds: [], confidence: null, error: null, done: false,
    }]);
    setQuery('');
    setIsLoading(true);
    const handleSSEEvent = (eventType, data) => handleTurnEvent(turnId, asked, eventType, data);

    const controller = new AbortController();

    fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        prompt: asked,
        // Earlier turns, so "When did they buy it?" can be resolved; empty for the first question
        history: conversation.slice(-MAX_TURNS),
        topK: 5,
        corpusId: mode === 'wiki' ? 'wiki' : corpusId,
        // Checked files scope the question; none checked = search every uploaded file
        fileIds: mode === 'files'
          ? uploadedFiles.filter((f) => selectedFileIds.has(f.fileId)).map((f) => f.fileId)
          : [],
      }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        // Event state lives outside the read loop: a large event (e.g. the final
        // "done" with evidence chunks) often arrives split across several reads.
        let eventType = '';
        let dataLines = [];

        const dispatch = () => {
          if (dataLines.length) {
            try {
              handleSSEEvent(eventType, JSON.parse(dataLines.join('\n')));
            } catch (err) { /* skip malformed */ }
          }
          eventType = '';
          dataLines = [];
        };

        const processLines = (lines) => {
          for (const rawLine of lines) {
            const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
            if (line === '') {
              dispatch(); // blank line ends an event
            } else if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).replace(/^ /, ''));
            }
          }
        };

        while (true) {
          const { done: readerDone, value } = await reader.read();
          if (readerDone) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          processLines(lines);
        }
        // Stream closed: handle anything left without a trailing blank line
        buffer += decoder.decode();
        processLines(buffer ? [buffer] : []);
        dispatch();
      })
      .catch((err) => {
        if (err.name !== 'AbortError') updateTurn(turnId, { error: err.message, status: null });
      })
      .finally(() => setIsLoading(false));

    eventSourceRef.current = { close: () => controller.abort() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, isLoading, mode, corpusId, uploadedFiles, selectedFileIds, conversation, updateTurn]);

  // Streamed events update only the turn they belong to; earlier turns keep what they showed
  const handleTurnEvent = (turnId, asked, eventType, data) => {
    switch (eventType) {
      case 'status':
        updateTurn(turnId, { status: { stage: data.stage, message: data.message } });
        break;
      case 'agent_step':
        updateTurn(turnId, (t) => ({ agentSteps: [...t.agentSteps, { agent: data.agent, detail: data.detail }] }));
        break;
      case 'chunk':
        updateTurn(turnId, (t) => ({ text: t.text + data.content }));
        break;
      case 'done': {
        const standalone = (data.standaloneQuery || asked).trim();
        updateTurn(turnId, (t) => ({
          // The full answer is authoritative (keeps every line break, even if a chunk was lost)
          text: data.answer || t.text,
          done: true,
          status: null,
          sources: data.sources || [],
          chunks: data.chunks || [],
          graphPaths: data.graphPaths || [],
          graphSeeds: data.graphSeeds || [],
          confidence: data.confidence ?? null,
          searchedAs: standalone.toLowerCase() !== asked.toLowerCase() ? standalone : '',
        }));
        setIsLoading(false);
        // The next question sees this one as the search understood it
        setConversation((prev) => [...prev, {
          question: standalone,
          answer: trimAnswer(data.answer || ''),
          entityKeys: data.entityKeys || [],
        }].slice(-MAX_TURNS));
        break;
      }
      case 'error':
        updateTurn(turnId, { error: data.message, status: null });
        setIsLoading(false);
        break;
      default:
        break;
    }
  };

  const startNewTopic = useCallback(() => {
    setConversation([]);
    turnCounter.current += 1;
    const id = turnCounter.current;
    setTurns((prev) => (prev.length && !prev[prev.length - 1].divider ? [...prev, { id, divider: true }] : prev));
    inputRef.current?.focus();
  }, []);

  // Enter sends, Shift+Enter starts a new line (like other chat apps)
  const onComposerKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent?.isComposing) {
      handleSearch(e);
    }
  }, [handleSearch]);

  // The question box grows with its text, up to a few lines
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [query]);

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

  return (
    <div className="app-shell">
      <div className="chat-scroll" ref={scrollRef} onScroll={onChatScroll}>
        <div className="app-container">
          <div className="app-top">
          <header className="app-header">
            <h1>GraphRAG Search</h1>
            <p className="brand-subtitle">
              Answers from your own files or a set of Wikipedia articles. Passages are ranked by how
              closely they match the question and by how central they are in the knowledge graph.
            </p>
          </header>

          <div className="mode-selector" role="group" aria-label="Search mode">
            <button
              type="button"
              className={`mode-button ${mode === 'files' ? 'active' : ''}`}
              aria-pressed={mode === 'files'}
              onClick={() => chooseMode('files')}
            >
              My files
            </button>
            <button
              type="button"
              className={`mode-button ${mode === 'wiki' ? 'active' : ''}`}
              aria-pressed={mode === 'wiki'}
              onClick={() => chooseMode('wiki')}
            >
              Wiki dump
            </button>
          </div>

          {mode === 'files' && (
            <section className="upload-panel" aria-label="Upload files">
              <div className="upload-heading">
                <div>
                  <h2>Ask questions about your files</h2>
                  <p>Answers in this mode use your uploaded files only.</p>
                </div>
                <span className="upload-format">TXT · MD · PDF · DOCX · MP3 · max 5 MB each</span>
              </div>
              <form className="upload-form" onSubmit={handleUpload}>
                <input
                  type="file"
                  name="files"
                  accept=".txt,.md,.pdf,.docx,.mp3,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,audio/mpeg"
                  multiple
                  disabled={uploading || creatingWorkspace || !corpusId}
                  aria-label="Choose text, Markdown, PDF, Word or MP3 files"
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
                  <div className="files-toolbar">
                    <label className="file-check">
                      <input
                        type="checkbox"
                        ref={selectAllRef}
                        checked={allSelected}
                        onChange={toggleAllFiles}
                        aria-label="Select all files"
                      />
                      <span className="files-heading">Indexed files ({uploadedFiles.length})</span>
                    </label>
                    <div className="file-actions">
                      <button
                        type="button"
                        className="file-action"
                        onClick={openMap}
                        title="Map entities, relationships and themes in the selected files (or all files), with the latest extracted facts"
                      >
                        Map{selectedCount ? ` (${selectedCount})` : ''}
                      </button>
                      <button
                        type="button"
                        className="file-action"
                        onClick={previewSelected}
                        disabled={!selectedCount}
                      >
                        Preview{selectedCount ? ` (${selectedCount})` : ''}
                      </button>
                      <button
                        type="button"
                        className="file-action file-action-danger"
                        onClick={deleteSelected}
                        disabled={!selectedCount || deletingFiles}
                      >
                        {deletingFiles ? 'Deleting…' : `Delete${selectedCount ? ` (${selectedCount})` : ''}`}
                      </button>
                    </div>
                  </div>
                  <ul>
                    {uploadedFiles.map((file) => (
                      <li key={file.fileId || file.filename}>
                        <label className="file-check">
                          <input
                            type="checkbox"
                            checked={selectedFileIds.has(file.fileId)}
                            onChange={() => toggleFileSelected(file.fileId)}
                            aria-label={`Select ${file.filename}`}
                          />
                          <span className="file-name">{file.filename}</span>
                        </label>
                        <span className="file-status">{file.chunkCount ? `${file.chunkCount} chunks · ` : ''}Ready</span>
                      </li>
                    ))}
                  </ul>
                  <p className="file-scope">
                    {selectedCount
                      ? `Questions will search the ${selectedCount} selected file${selectedCount === 1 ? '' : 's'} only.`
                      : `No files selected: questions will search all ${uploadedFiles.length} files.`}
                  </p>
                  {fileActionError && <p className="upload-error">{fileActionError}</p>}
                  {previewIds.length > 0 && (
                    <div className="file-preview" aria-label="File preview" ref={previewRef}>
                      <div className="file-preview-header">
                        <h3>Preview</h3>
                        <button type="button" className="file-action" onClick={() => setPreviewIds([])}>
                          Close
                        </button>
                      </div>
                      {previewIds.map((id) => {
                        const file = uploadedFiles.find((f) => f.fileId === id);
                        const cached = previewCache[id];
                        return (
                          <div key={id} className="file-preview-item">
                            <h4>{file ? file.filename : id}</h4>
                            {cached?.loading && <span className="evidence-loading">Loading…</span>}
                            {cached?.error && <span className="evidence-error">{cached.error}</span>}
                            {cached?.content != null && <pre className="file-preview-text">{cached.content}</pre>}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {showMap && mapRequest && (
                    <div ref={mapRef} className="corpus-map-anchor">
                      <CorpusMap
                        apiBase={API_BASE}
                        corpusId={corpusId}
                        files={uploadedFiles}
                        scopeFileIds={uploadedFiles.filter((f) => selectedFileIds.has(f.fileId)).map((f) => f.fileId)}
                        request={mapRequest}
                        onClose={closeMap}
                        hoverHighlight={hoverHighlight}
                        pinnedHighlight={pinnedHighlight}
                        onClearPinned={() => setPinnedHighlight(null)}
                        onBackToEvidence={backToEvidence}
                      />
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          </div>

          <section className="thread" aria-label="Conversation">
            {turns.map((turn, i) => {
              const latest = i === turns.length - 1;
              if (turn.divider) {
                return (
                  <div key={turn.id} className="thread-divider" ref={latest ? latestTurnRef : undefined}>
                    <span>New topic: the next question starts fresh</span>
                  </div>
                );
              }
              return (
                <div key={turn.id} ref={latest ? latestTurnRef : undefined} className="thread-item">
                  <ChatTurn
                    turn={turn}
                    mode={mode}
                    mapOpen={mapOpen}
                    pinnedHighlight={pinnedHighlight}
                    hoverProps={hoverProps}
                    toggleShowInMap={toggleShowInMap}
                    sourceCache={sourceCache}
                    fetchChunkSource={fetchChunkSource}
                  />
                </div>
              );
            })}
          </section>

          {!turns.some((t) => !t.divider) && (
            <div className="empty-state">
              <p>
                {mode === 'files' && !uploadedFiles.length
                  ? 'Upload a text, Markdown, PDF, Word or MP3 file above, then ask a question about it below.'
                  : mode === 'files'
                    ? 'Ask a question about your files below. Tick files above to search only those.'
                    : 'Ask a question about the articles in the wiki dump below.'}
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          {!atBottom && turns.length > 0 && (
            <button type="button" className="jump-latest" onClick={scrollToLatest} aria-label="Scroll to the latest answer">
              ↓
            </button>
          )}
          <form className="search-form" onSubmit={handleSearch}>
            <div className="search-bar">
              <textarea
                ref={inputRef}
                rows={1}
                className="search-input"
                aria-label="Question"
                placeholder={mode === 'wiki'
                  ? 'Ask about the wiki dump…'
                  : 'Ask a question about your uploaded files…'}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onComposerKeyDown}
                autoFocus
              />
              <button
                type="submit"
                className="search-button"
                disabled={isLoading || !query.trim() || (mode === 'files' && !uploadedFiles.length)}
              >
                {isLoading ? 'Searching…' : 'Search'}
              </button>
            </div>
            <p className="chat-context">
              {conversation.length > 0 ? (
                <>
                  Follow-up questions can refer to your last {conversation.length === 1
                    ? 'question' : `${conversation.length} questions`}.
                  <button
                    type="button"
                    className="file-action chat-reset"
                    onClick={startNewTopic}
                    disabled={isLoading}
                  >
                    New topic
                  </button>
                </>
              ) : (
                <span>Enter to search, Shift+Enter for a new line.</span>
              )}
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
