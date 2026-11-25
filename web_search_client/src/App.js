import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import './App.css';
import ChatSidebar from './ChatSidebar';
import ChatTurn from './ChatTurn';
import CorpusMap from './CorpusMap';
import { API_BASE } from './config';
import { deleteChat as deleteStoredChat, loadChats, saveChat } from './chatStore';

// Follow-ups: the last few questions and the start of their answers go with each search, so
// the query processor can rewrite "When did they buy it?" (it uses at most 3 turns)
const MAX_TURNS = 3;
const MAX_ANSWER_CHARS = 300;
const MAX_TITLE_CHARS = 60;
const INTERRUPTED = 'This answer was interrupted: the page was closed or reloaded before it finished.';
const ENDED_EARLY = 'The search ended before an answer arrived. Try asking again.';

function trimAnswer(text) {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (flat.length <= MAX_ANSWER_CHARS) return flat;
  const cut = flat.slice(0, MAX_ANSWER_CHARS);
  const space = cut.lastIndexOf(' ');
  return `${space > MAX_ANSWER_CHARS / 2 ? cut.slice(0, space) : cut}…`;
}

let idCounter = 0;
function newId(prefix) {
  idCounter += 1;
  const random = window.crypto?.randomUUID ? window.crypto.randomUUID() : Math.random().toString(36).slice(2);
  return `${prefix}-${Date.now().toString(36)}-${idCounter}-${random.slice(0, 8)}`;
}

function titleFrom(question) {
  const flat = question.replace(/\s+/g, ' ').trim();
  return flat.length > MAX_TITLE_CHARS ? `${flat.slice(0, MAX_TITLE_CHARS - 1)}…` : flat;
}

/**
 * A chat, as kept in memory and saved in IndexedDB (chatStore.js):
 *   { id, title, mode: 'files' | 'wiki', corpusId, fileIds (ticked when last asked),
 *     createdAt, updatedAt, turns: [turn | { id, divider: true }],
 *     context: [{ question, answer, entityKeys }] (the last turns sent with a follow-up) }
 * A turn is what ChatTurn shows: { id, question, searchedAs, status, agentSteps, text,
 *   sources, chunks, graphPaths, graphSeeds, confidence, error, done }.
 */
function emptyChat(id, mode) {
  const now = Date.now();
  return { id, title: '', mode, corpusId: '', fileIds: [], createdAt: now, updatedAt: now, turns: [], context: [] };
}

// A turn still being written when the page closed can't be resumed: say so instead of
// showing it as if it were still running
function markInterrupted(chat) {
  if (!chat.turns.some((t) => !t.divider && !t.done && !t.error)) return chat;
  return {
    ...chat,
    turns: chat.turns.map((t) => (t.divider || t.done || t.error ? t : { ...t, status: null, error: INTERRUPTED })),
  };
}

function patchTurn(chat, turnId, change) {
  return {
    ...chat,
    turns: chat.turns.map((t) => (t.id === turnId
      ? { ...t, ...(typeof change === 'function' ? change(t) : change) }
      : t)),
  };
}

// The sidebar is drawn from a string of what it shows (id, title, mode, time per chat), so a
// streamed token, which changes none of these, doesn't redraw it
const FIELD_SEP = '\u0001';
const ROW_SEP = '\u0002';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('files');
  const [corpusId, setCorpusId] = useState(
    () => window.localStorage.getItem('graphrag-corpus-id') || ''
  );
  const [corpusError, setCorpusError] = useState(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [filesLoaded, setFilesLoaded] = useState(false);
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

  // Chats by id. The open chat may be a new one that has no question yet and isn't saved.
  const [chats, setChats] = useState({});
  const [activeId, setActiveId] = useState(() => newId('chat'));
  const [chatsLoaded, setChatsLoaded] = useState(false);
  // The one search that runs at a time (the local model answers one question at once):
  // { chatId, turnId }. It keeps writing into its chat while another chat is open.
  const [running, setRunning] = useState(null);
  const abortRef = useRef(null);
  const savedRef = useRef(new Map()); // chat id -> the chat object last written to IndexedDB
  // Latest values for callbacks that must stay the same function (memoised sidebar rows)
  const chatsRef = useRef(chats);
  const activeIdRef = useRef(activeId);
  const runningRef = useRef(running);
  chatsRef.current = chats;
  activeIdRef.current = activeId;
  runningRef.current = running;
  const [sidebarOpen, setSidebarOpen] = useState(false); // narrow screens: chats drawer
  const scrollRef = useRef(null);       // the scrolling chat area
  const latestTurnRef = useRef(null);
  const shownChatRef = useRef(activeId);
  const [atBottom, setAtBottom] = useState(true);
  const inputRef = useRef(null);

  const activeChat = chats[activeId] || null;
  const turns = activeChat ? activeChat.turns : [];
  const context = activeChat ? activeChat.context : [];
  const isLoading = Boolean(running);
  const runningElsewhere = Boolean(running && running.chatId !== activeId);

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
          setFilesLoaded(true);
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

  // Saved chats, from this browser's database
  useEffect(() => {
    let active = true;
    loadChats()
      .then((list) => {
        if (!active) return;
        const loaded = {};
        list.forEach((stored) => {
          const chat = markInterrupted(stored);
          loaded[chat.id] = chat;
          if (chat === stored) savedRef.current.set(chat.id, chat);
        });
        // A chat started before loading finished wins over nothing: ids never collide
        setChats((prev) => ({ ...loaded, ...prev }));
      })
      .catch((err) => console.warn('Could not load saved chats', err))
      .finally(() => { if (active) setChatsLoaded(true); });
    return () => { active = false; };
  }, []);

  // Save chats that changed. While an answer streams in, its chat is saved when the question
  // is added (so a reload keeps the question) but not on every token; it is saved again once
  // the answer is done. Chats without a question are not saved.
  useEffect(() => {
    if (!chatsLoaded) return;
    Object.values(chats).forEach((chat) => {
      if (!chat.turns.length) return;
      const saved = savedRef.current.get(chat.id);
      if (saved === chat) return;
      if (running?.chatId === chat.id && saved && saved.turns.length === chat.turns.length) return;
      savedRef.current.set(chat.id, chat);
      saveChat(chat).catch((err) => console.warn('Could not save chat', err));
    });
  }, [chats, running, chatsLoaded]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const updateChat = useCallback((chatId, change) => {
    setChats((prev) => (prev[chatId]
      ? { ...prev, [chatId]: { ...prev[chatId], ...change(prev[chatId]) } }
      : prev));
  }, []);

  const updateTurn = useCallback((chatId, turnId, change) => {
    setChats((prev) => (prev[chatId] ? { ...prev, [chatId]: patchTurn(prev[chatId], turnId, change) } : prev));
  }, []);

  const mapOpen = showMap && mode === 'files';

  // Opening a chat shows its latest turn; a new question scrolls up to the top of the view,
  // so its answer is written below it
  useEffect(() => {
    if (shownChatRef.current !== activeId) {
      shownChatRef.current = activeId;
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
      return;
    }
    if (latestTurnRef.current) latestTurnRef.current.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }, [turns.length, activeId]);

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

  // Hover handlers that highlight something in the open corpus map (the same function until
  // the map opens or closes, so finished turns aren't redrawn)
  const hoverProps = useCallback((highlight) => (mapOpen
    ? {
      onMouseEnter: () => setHoverHighlight(highlight),
      onMouseLeave: () => setHoverHighlight(null),
      onFocus: () => setHoverHighlight(highlight),
      onBlur: () => setHoverHighlight(null),
    }
    : {}), [mapOpen]);

  // Files in the workspace right now: sources and evidence from deleted files say so
  const availableFileIds = useMemo(
    () => (mode === 'files' && filesLoaded ? new Set(uploadedFiles.map((f) => f.fileId)) : null),
    [mode, filesLoaded, uploadedFiles],
  );

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

  // Each chat searches one corpus. Switching with questions already asked starts a new chat
  // for the other corpus; the current one stays in the sidebar (and keeps running if it is).
  const chooseMode = useCallback((nextMode) => {
    if (nextMode === mode) return;
    if (chats[activeId]?.turns.length) setActiveId(newId('chat'));
    setPinnedHighlight(null);
    setHoverHighlight(null);
    setMode(nextMode);
  }, [mode, chats, activeId]);

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
    // One search at a time, across all chats
    if (!query.trim() || running || (mode === 'files' && !uploadedFiles.length)) return;

    const asked = query.trim();
    const chatId = activeId;
    const turnId = newId('turn');
    const fileIds = mode === 'files'
      ? uploadedFiles.filter((f) => selectedFileIds.has(f.fileId)).map((f) => f.fileId)
      : [];
    // Earlier turns of this chat, so "When did they buy it?" can be resolved; empty at first
    const history = (chats[chatId]?.context || []).slice(-MAX_TURNS);
    setChats((prev) => {
      const chat = prev[chatId] || emptyChat(chatId, mode);
      return {
        ...prev,
        [chatId]: {
          ...chat,
          title: chat.title || titleFrom(asked),
          mode,
          corpusId: mode === 'wiki' ? 'wiki' : corpusId,
          fileIds,
          updatedAt: Date.now(),
          turns: [...chat.turns, {
            id: turnId, question: asked, searchedAs: '', status: null, agentSteps: [], text: '',
            sources: [], chunks: [], graphPaths: [], graphSeeds: [], confidence: null, error: null, done: false,
          }],
        },
      };
    });
    setQuery('');
    setRunning({ chatId, turnId });
    const handleSSEEvent = (eventType, data) => handleTurnEvent(chatId, turnId, asked, eventType, data);

    const controller = new AbortController();
    abortRef.current = controller;

    fetch(`${API_BASE}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        prompt: asked,
        history,
        topK: 5,
        corpusId: mode === 'wiki' ? 'wiki' : corpusId,
        // Checked files scope the question; none checked = search every uploaded file
        fileIds,
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
        if (err.name !== 'AbortError') updateTurn(chatId, turnId, { error: err.message, status: null });
      })
      .finally(() => {
        // A stream that closed without "done" or "error" leaves no answer: say so
        updateTurn(chatId, turnId, (t) => (t.done || t.error ? {} : { status: null, error: ENDED_EARLY }));
        setRunning((r) => (r && r.turnId === turnId ? null : r));
        if (abortRef.current === controller) abortRef.current = null;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, running, mode, activeId, chats, corpusId, uploadedFiles, selectedFileIds, updateTurn]);

  // Streamed events update only the turn they belong to, in its own chat (which may no longer
  // be the one on screen); earlier turns keep what they showed
  const handleTurnEvent = (chatId, turnId, asked, eventType, data) => {
    switch (eventType) {
      case 'status':
        updateTurn(chatId, turnId, { status: { stage: data.stage, message: data.message } });
        break;
      case 'agent_step':
        updateTurn(chatId, turnId, (t) => ({ agentSteps: [...t.agentSteps, { agent: data.agent, detail: data.detail }] }));
        break;
      case 'chunk':
        updateTurn(chatId, turnId, (t) => ({ text: t.text + data.content }));
        break;
      case 'done': {
        const standalone = (data.standaloneQuery || asked).trim();
        updateChat(chatId, (chat) => ({
          ...patchTurn(chat, turnId, (t) => ({
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
          })),
          // The next question in this chat sees this one as the search understood it
          context: [...chat.context, {
            question: standalone,
            answer: trimAnswer(data.answer || ''),
            entityKeys: data.entityKeys || [],
          }].slice(-MAX_TURNS),
          updatedAt: Date.now(),
        }));
        setRunning((r) => (r && r.turnId === turnId ? null : r));
        break;
      }
      case 'error':
        updateTurn(chatId, turnId, { error: data.message, status: null });
        setRunning((r) => (r && r.turnId === turnId ? null : r));
        break;
      default:
        break;
    }
  };

  const startNewTopic = useCallback(() => {
    updateChat(activeId, (chat) => ({
      context: [],
      turns: chat.turns.length && !chat.turns[chat.turns.length - 1].divider
        ? [...chat.turns, { id: newId('turn'), divider: true }]
        : chat.turns,
    }));
    inputRef.current?.focus();
  }, [activeId, updateChat]);

  // --- Chats (sidebar). These keep the same function between renders, reading the latest
  // state from refs, so the memoised sidebar rows aren't redrawn by a streaming answer.
  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  const openChat = useCallback((id) => {
    const chat = chatsRef.current[id];
    setSidebarOpen(false);
    if (!chat || id === activeIdRef.current) return;
    setActiveId(id);
    setPinnedHighlight(null);
    setHoverHighlight(null);
    setMode(chat.mode);
    // Back to the files this chat last searched (none ticked = all files)
    if (chat.mode === 'files') setSelectedFileIds(new Set(chat.fileIds || []));
  }, []);

  const startNewChat = useCallback(() => {
    setSidebarOpen(false);
    const current = chatsRef.current[activeIdRef.current];
    if (current && current.turns.length) {
      setActiveId(newId('chat'));
      setQuery('');
      setPinnedHighlight(null);
      setHoverHighlight(null);
    }
    inputRef.current?.focus();
  }, []);

  const renameChat = useCallback((id, title) => {
    setChats((prev) => (prev[id] ? { ...prev, [id]: { ...prev[id], title } } : prev));
  }, []);

  const removeChat = useCallback((id) => {
    const chat = chatsRef.current[id];
    if (!chat) return;
    if (!window.confirm(`Delete the chat “${chat.title}”? This can't be undone.`)) return;
    if (runningRef.current?.chatId === id) {
      abortRef.current?.abort();
      setRunning(null);
    }
    setChats((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    savedRef.current.delete(id);
    deleteStoredChat(id).catch((err) => console.warn('Could not delete chat', err));
    if (activeIdRef.current === id) setActiveId(newId('chat'));
  }, []);

  // Sidebar rows, newest first; recomputed only when a title, mode or time changes
  const sidebarKey = Object.values(chats)
    .filter((c) => c.turns.length)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .map((c) => [c.id, c.title, c.mode, c.updatedAt].join(FIELD_SEP))
    .join(ROW_SEP);
  const sidebarItems = useMemo(() => (sidebarKey
    ? sidebarKey.split(ROW_SEP).map((row) => {
      const [id, title, chatMode, updatedAt] = row.split(FIELD_SEP);
      return { id, title, mode: chatMode, updatedAt: Number(updatedAt) };
    })
    : []), [sidebarKey]);

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
            : { loading: false, content: null, error: 'This source is no longer available. Its file may have been deleted.' },
        }));
      })
      .catch((err) => {
        setSourceCache((prev) => ({ ...prev, [uri]: { loading: false, content: null, error: err.message } }));
      });
  }, []);

  const runningTitle = running ? chats[running.chatId]?.title || 'another chat' : '';

  return (
    <div className="app-shell">
      <ChatSidebar
        items={sidebarItems}
        activeId={activeId}
        runningId={running?.chatId || null}
        open={sidebarOpen}
        onNew={startNewChat}
        onSelect={openChat}
        onRename={renameChat}
        onDelete={removeChat}
        onClose={closeSidebar}
      />
      {sidebarOpen && <div className="sidebar-backdrop" onClick={closeSidebar} aria-hidden="true" />}
      <div className="main-pane">
      <div className="chat-scroll" ref={scrollRef} onScroll={onChatScroll}>
        <div className="app-container">
          <div className="app-top">
          <header className="app-header">
            <button type="button" className="chats-toggle" onClick={() => setSidebarOpen(true)}>
              ☰ Chats
            </button>
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
                    availableFileIds={availableFileIds}
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
            {runningElsewhere && (
              <p className="busy-note" role="status">
                “{runningTitle}” is still being answered. You can type your question now and send it
                when that answer finishes.
                <button type="button" className="file-action" onClick={() => openChat(running.chatId)}>
                  Open it
                </button>
              </p>
            )}
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
                {runningElsewhere ? 'Waiting…' : isLoading ? 'Searching…' : 'Search'}
              </button>
            </div>
            <p className="chat-context">
              {context.length > 0 ? (
                <>
                  Follow-up questions can refer to your last {context.length === 1
                    ? 'question' : `${context.length} questions`}.
                  <button
                    type="button"
                    className="file-action chat-reset"
                    onClick={startNewTopic}
                    disabled={running?.chatId === activeId}
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
    </div>
  );
}

export default App;
