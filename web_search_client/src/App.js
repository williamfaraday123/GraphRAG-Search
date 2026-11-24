import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';
import CorpusMap from './CorpusMap';
import Markdown, { parseOptionChecks, renderInline, splitAnswer } from './Markdown';
import { API_BASE } from './config';
import { formatSourceLabel, sourcePageHref, sourcePath } from './sourceLabel';

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
  const [graphPaths, setGraphPaths] = useState([]);
  const [graphSeeds, setGraphSeeds] = useState([]);
  const [scrollToMap, setScrollToMap] = useState(0);
  const [mapRequest, setMapRequest] = useState(null); // { id, fileIds }: one per Map click
  const mapRef = useRef(null);
  const [scrollToPreview, setScrollToPreview] = useState(0);
  const previewRef = useRef(null);
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
  // Earlier turns sent with each follow-up: {question, answer (trimmed), entityKeys}, oldest first
  const [conversation, setConversation] = useState([]);
  const [searchedAs, setSearchedAs] = useState(''); // the rewritten question, when it was rewritten
  const askedRef = useRef('');
  const eventSourceRef = useRef(null);
  const resultRef = useRef(null);
  const answerRef = useRef(null);

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
    setHoverHighlight(null);
    setPinnedHighlight(null);
    setGraphPaths([]);
    setGraphSeeds([]);
    setSearchedAs('');
  }, []);

  const mapOpen = showMap && mode === 'files';

  // After "Show in map" the map may only just be mounting, so scroll once it exists
  useEffect(() => {
    if (scrollToMap && mapRef.current) {
      mapRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [scrollToMap]);

  // Once the answer is complete, bring it into view (jsdom has no scrollIntoView)
  useEffect(() => {
    if (done && answerRef.current) {
      answerRef.current.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    }
  }, [done]);

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
  const chunkHighlight = (chunkId) => ({ key: `chunk:${chunkId}`, kind: 'chunk', chunkId, anchorId: `evidence-${chunkId}` });
  // Shared facts (file-similarity answers) or a reasoning path; the map banner names which
  const pathLabel = graphPaths.length && graphPaths.every((p) => p.kind === 'shared') ? 'Shared facts' : 'Reasoning path';
  const pathHighlight = (index) => (index === 'all'
    ? { key: 'path:all', kind: 'path', label: pathLabel, facts: graphPaths.flatMap((p) => p.facts), anchorId: 'graph-reasoning' }
    : { key: `path:${index}`, kind: 'path', label: pathLabel, facts: graphPaths[index].facts, anchorId: `graph-path-${index}` });

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
    setMode(nextMode);
    resetState();
    setConversation([]); // earlier questions were about the other corpus
  }, [resetState]);

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

    resetState();
    setIsLoading(true);
    askedRef.current = query.trim();

    const controller = new AbortController();

    fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        prompt: query.trim(),
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
        if (err.name !== 'AbortError') setError(err.message);
      })
      .finally(() => setIsLoading(false));

    eventSourceRef.current = { close: () => controller.abort() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, isLoading, mode, corpusId, uploadedFiles, selectedFileIds, resetState, conversation]);

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
        // The full answer is authoritative (keeps every line break, even if a chunk was lost)
        if (data.answer) setStreamedText(data.answer);
        setDone(true);
        setStatus(null);
        setIsLoading(false);
        setSources(data.sources || []);
        setChunks(data.chunks || []);
        setGraphPaths(data.graphPaths || []);
        setGraphSeeds(data.graphSeeds || []);
        setConfidence(data.confidence);
        {
          const asked = askedRef.current;
          const standalone = (data.standaloneQuery || asked).trim();
          setSearchedAs(standalone.toLowerCase() !== asked.toLowerCase() ? standalone : '');
          // The next question sees this one as the search understood it
          setConversation((prev) => [...prev, {
            question: standalone,
            answer: trimAnswer(data.answer || ''),
            entityKeys: data.entityKeys || [],
          }].slice(-MAX_TURNS));
        }
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

  const agentLabels = {
    rewriter: 'Rewriter',
    planner: 'Planner',
    retriever: 'Retriever',
    analyst: 'Critic',
    synthesizer: 'Synthesizer',
  };

  // The query processor's status stages, shown as numbered steps. A second retrieval round
  // (after the critic finds a gap) goes back to step 2.
  const stageSteps = ['Plan', 'Retrieve', 'Check', 'Write'];
  const stageIndex = {
    PLANNING: 0, RETRIEVING: 1, ITERATING: 1, ANALYZING: 2, SYNTHESIZING: 3,
  }[status?.stage] ?? -1;

  return (
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
            {isLoading ? 'Searching…' : 'Search'}
          </button>
        </div>
        {conversation.length > 0 && (
          <p className="chat-context">
            {searchedAs && <span>Searched as “{searchedAs}”. </span>}
            Follow-up questions can refer to your last {conversation.length === 1
              ? 'question' : `${conversation.length} questions`}.
            <button
              type="button"
              className="file-action chat-reset"
              onClick={() => { setConversation([]); setSearchedAs(''); }}
              disabled={isLoading}
            >
              New topic
            </button>
          </p>
        )}
      </form>

      {status && (
        <div className="progress" role="status">
          {stageIndex >= 0 && (
            <ol className="progress-steps">
              {stageSteps.map((label, i) => (
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

      {agentSteps.length > 0 && (
        <div className="agent-trace">
          <h3 className="agent-trace-label">Search log</h3>
          <ul className="agent-trace-list">
            {agentSteps.map((step, i) => (
              <li key={i} className="agent-trace-item">
                <span className="agent-trace-name">{agentLabels[step.agent] || step.agent}</span>
                {/* The critic's detail is the model's own Markdown: show its bold, not its asterisks */}
                <span className="agent-trace-detail">{renderInline((step.detail || '').replace(/\s+/g, ' '), `trace-${i}`)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="error-bar" role="alert">
          {error}
        </div>
      )}
      </div>

      {(streamedText || done) && (
        <div className="results" ref={resultRef}>
          <div className="results-main">
          <div className="answer-section" ref={answerRef}>
            <h2 className="answer-label">Answer</h2>
            {(() => {
              // The model writes "### Checked evidence" then "### Answer" (it reasons before it
              // answers). Show the answer first. Multiple choice: a summary, then every option
              // with its verdict. Otherwise: the checks one click away.
              const { answer, checks } = splitAnswer(streamedText);
              const options = done ? parseOptionChecks(checks) : null;
              if (options) {
                const supported = options.filter((o) => o.supported).length;
                return (
                  <>
                    <div className="answer-text answer-summary">
                      <Markdown text={answer} />
                    </div>
                    <section className="option-breakdown" aria-labelledby="option-breakdown-title">
                      <h3 id="option-breakdown-title" className="option-breakdown-title">
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
                            {o.reason && <p className="option-reason">{renderInline(o.reason, `r-${o.marker}`)}</p>}
                            {o.evidence && <p className="option-evidence">{renderInline(o.evidence, `e-${o.marker}`)}</p>}
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

          {done && confidence !== null && (
            <p className="confidence">
              Confidence {(confidence * 100).toFixed(0)}%
            </p>
          )}

          {sources.length > 0 && (
            <div className="sources-section">
              <h3>Sources</h3>
              <ul className="sources-list">
                {sources.map((src, i) => (
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
                ))}
              </ul>
            </div>
          )}
          </div>

          {done && (graphPaths.length > 0 || chunks.length > 0) && (
          <aside className="results-side" aria-label="Evidence">
          {graphPaths.length > 0 && (() => {
            const sharedFacts = pathLabel === 'Shared facts';
            return (
            <div className="graph-reasoning" id="graph-reasoning" aria-labelledby="graph-reasoning-title">
              <div className="graph-reasoning-header">
                <h3 className="evidence-label" id="graph-reasoning-title">
                  {sharedFacts ? 'Shared facts' : 'Reasoning path'}
                </h3>
                {mode === 'files' && (
                  <button
                    type="button"
                    className="evidence-toggle"
                    aria-pressed={pinnedHighlight?.key === 'path:all'}
                    onClick={() => toggleShowInMap(pathHighlight('all'))}
                  >
                    {pinnedHighlight?.key === 'path:all' ? 'Stop highlighting in map' : 'Show all in map'}
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
                  const pinned = pinnedHighlight?.key === `path:${i}`;
                  return (
                    <li
                      key={i}
                      id={`graph-path-${i}`}
                      className={`graph-path ${pinned ? 'in-map' : ''}`}
                      {...(mode === 'files' ? hoverProps(pathHighlight(i)) : {})}
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
                          onClick={() => toggleShowInMap(pathHighlight(i))}
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
                  const mappable = mode === 'files' && Boolean(c.chunkId);
                  const pinned = mappable && pinnedHighlight?.key === `chunk:${c.chunkId}`;
                  return (
                    <li
                      key={c.chunkId || i}
                      id={c.chunkId ? `evidence-${c.chunkId}` : undefined}
                      className={`evidence-item ${pinned ? 'in-map' : ''}`}
                      {...(mappable ? hoverProps(chunkHighlight(c.chunkId)) : {})}
                    >
                      <div className="evidence-header">
                        <span className="evidence-source">{formatSourceLabel(c.source)}</span>
                        {c.foundByGraph && (
                          <span className="evidence-badge-graph"
                            title="Vector search did not return this chunk; a knowledge-graph fact path led to it">
                            Found via knowledge graph
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
                        {c.source && (
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
                            onClick={() => toggleShowInMap(chunkHighlight(c.chunkId))}
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

      {!streamedText && !isLoading && !error && (
        <div className="empty-state">
          <p>
            {mode === 'files' && !uploadedFiles.length
              ? 'Upload a text, Markdown, PDF, Word or MP3 file above, then ask a question about it.'
              : mode === 'files'
                ? 'Ask a question about your files. Tick files above to search only those.'
                : 'Ask a question about the articles in the wiki dump.'}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
