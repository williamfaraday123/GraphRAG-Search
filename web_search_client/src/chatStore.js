/**
 * Saved chats, in this browser's own database (IndexedDB), like Ollama's app keeps chats on
 * your machine. Nothing is sent to the server: the query processor is stateless and gets each
 * chat's last turns with every question. Chats belong to this browser profile; clearing the
 * site's data deletes them.
 *
 * One record per chat: { id, title, mode, corpusId, fileIds, createdAt, updatedAt, turns,
 * context } (shapes in App.js). Where IndexedDB is unavailable (some private windows, tests),
 * chats are kept in memory for the session instead.
 */
const DB_NAME = 'graphrag-search';
const DB_VERSION = 1;
const STORE = 'chats';

let dbPromise = null;
const memory = new Map(); // fallback when IndexedDB can't be opened

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      resolve(null);
      return;
    }
    let request;
    try {
      request = window.indexedDB.open(DB_NAME, DB_VERSION);
    } catch (err) {
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' }).createIndex('updatedAt', 'updatedAt');
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
  return dbPromise;
}

function run(db, mode, action) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const result = action(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(result?.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

/** Every saved chat, most recently updated first. */
export async function loadChats() {
  const db = await openDb();
  const chats = db ? (await run(db, 'readonly', (store) => store.getAll())) || [] : [...memory.values()];
  return chats.sort((a, b) => b.updatedAt - a.updatedAt);
}

export async function saveChat(chat) {
  const db = await openDb();
  if (!db) {
    memory.set(chat.id, chat);
    return;
  }
  await run(db, 'readwrite', (store) => store.put(chat));
}

export async function deleteChat(id) {
  const db = await openDb();
  if (!db) {
    memory.delete(id);
    return;
  }
  await run(db, 'readwrite', (store) => store.delete(id));
}

/** Tests only: forget the in-memory chats. */
export function resetMemoryStore() {
  memory.clear();
}
