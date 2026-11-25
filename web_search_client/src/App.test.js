import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from './App';
import { resetMemoryStore } from './chatStore';
import { scrollInChat } from './scrollInChat';

// jsdom has no layout: record what the chat area was asked to scroll to
jest.mock('./scrollInChat', () => ({ scrollInChat: jest.fn() }));
const scrolledTo = () => scrollInChat.mock.calls.map(([, el]) => el).filter(Boolean);

const CORPUS = '00000000-0000-4000-8000-000000000001';
const FILES = [
  { fileId: '11111111-1111-4111-8111-111111111111', filename: 'cinema.txt' },
  { fileId: '22222222-2222-4222-8222-222222222222', filename: 'tea.txt' },
];

const [CINEMA, TEA] = FILES.map((f) => f.fileId);
const MAP = {
  stats: { files: 2, chunks: 3, entities: 3, relationships: 2, clusters: 2 },
  files: FILES,
  entities: [
    { id: 'hollywood', label: 'Hollywood', type: 'name', mentions: 4, chunkCount: 2, fileIds: [CINEMA, TEA],
      chunkIds: [`${CINEMA}:0`, `${TEA}:0`], x: 0.5, y: 0.5 },
    { id: 'europe', label: 'Europe', type: 'name', mentions: 1, chunkCount: 1, fileIds: [CINEMA],
      chunkIds: [`${CINEMA}:0`], x: 0, y: 0 },
    { id: 'special effects', label: 'special effects', type: 'concept', mentions: 2, chunkCount: 2,
      fileIds: [CINEMA, TEA], chunkIds: [`${CINEMA}:1`, `${TEA}:0`], x: 1, y: 1 },
  ],
  relationships: [
    { source: 'europe', target: 'hollywood', weight: 1, fileIds: [CINEMA],
      examples: [{ chunkId: `${CINEMA}:0`, filename: 'cinema.txt', sentence: 'Hollywood overtook Europe.' }] },
    { source: 'hollywood', target: 'special effects', weight: 2, fileIds: [CINEMA, TEA],
      examples: [{ chunkId: `${TEA}:0`, filename: 'tea.txt', sentence: 'Hollywood had better special effects.' }] },
  ],
  clusters: [
    { id: 1, label: 'Hollywood, special effects', entityIds: ['hollywood', 'special effects'],
      files: [{ fileId: CINEMA, filename: 'cinema.txt', chunkCount: 1 }, { fileId: TEA, filename: 'tea.txt', chunkCount: 1 }],
      chunks: [
        { chunkId: `${CINEMA}:1`, fileId: CINEMA, filename: 'cinema.txt', chunkIndex: 1, snippet: 'Effects drew crowds.' },
        { chunkId: `${TEA}:0`, fileId: TEA, filename: 'tea.txt', chunkIndex: 0, snippet: 'Hollywood had better special effects.' },
      ] },
    { id: 2, label: 'Europe', entityIds: ['europe'],
      files: [{ fileId: CINEMA, filename: 'cinema.txt', chunkCount: 1 }],
      chunks: [{ chunkId: `${CINEMA}:0`, fileId: CINEMA, filename: 'cinema.txt', chunkIndex: 0, snippet: 'Hollywood overtook Europe.' }] },
  ],
  chunks: {
    [`${CINEMA}:0`]: { fileId: CINEMA, filename: 'cinema.txt', chunkIndex: 0, snippet: 'Hollywood overtook Europe.' },
    [`${CINEMA}:1`]: { fileId: CINEMA, filename: 'cinema.txt', chunkIndex: 1, snippet: 'Effects drew crowds.' },
    [`${TEA}:0`]: { fileId: TEA, filename: 'tea.txt', chunkIndex: 0, snippet: 'Hollywood had better special effects.' },
  },
};

const json = (data, ok = true, status = 200) =>
  Promise.resolve({ ok, status, json: () => Promise.resolve(data) });

// Routes fetch calls by method + URL so each test can inspect what the UI sent. CORPUS is the
// workspace the browser already had; each new chat's first upload creates another (NEW_CORPUS_n).
const newCorpus = (n) => `00000000-0000-4000-8000-0000000001${String(n).padStart(2, '0')}`;
let workspaces;
function mockApi({ files = FILES } = {}) {
  workspaces = { [CORPUS]: files };
  global.fetch = jest.fn((url, options = {}) => {
    const method = options.method || 'GET';
    if (url.endsWith('/api/v1/corpora') && method === 'POST') {
      const id = newCorpus(Object.keys(workspaces).length);
      workspaces[id] = [];
      return json({ corpusId: id });
    }
    const filesOf = url.match(/\/corpora\/([^/]+)\/files$/);
    if (filesOf && method === 'GET') return json(workspaces[filesOf[1]] || []);
    if (filesOf && method === 'POST') {
      const added = [...options.body.getAll('files')].map((f, i) => ({
        fileId: `${filesOf[1].slice(0, 30)}${String(workspaces[filesOf[1]].length + i).padStart(6, '0')}`,
        filename: f.name, success: true, chunkCount: 1,
      }));
      workspaces[filesOf[1]] = [...workspaces[filesOf[1]], ...added];
      return json({ files: added });
    }
    if (url.endsWith('/api/v1/search')) {
      return Promise.resolve({
        ok: true,
        body: { getReader: () => ({ read: () => Promise.resolve({ done: true }) }) },
      });
    }
    const match = url.match(/\/files\/([^/]+)$/);
    if (match && method === 'GET') {
      const file = files.find((f) => f.fileId === match[1]);
      return json({ fileId: file.fileId, filename: file.filename, content: `Full text of ${file.filename}` });
    }
    if (match && method === 'DELETE') return json({ success: true, message: 'Deleted' });
    if (url.includes(`/corpora/${CORPUS}/map`)) return json(MAP);
    return json({}, false, 404);
  });
}

async function openMyFiles() {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'My files' }));
  await screen.findByLabelText('Select cinema.txt');
}

beforeEach(() => {
  window.localStorage.setItem('graphrag-corpus-id', CORPUS);
  resetMemoryStore(); // jsdom has no IndexedDB: chats are kept in memory, fresh per test
  mockApi();
  scrollInChat.mockClear();
});

test('opens in My files mode, listed first, and switches to the wiki', async () => {
  global.fetch = jest.fn(() => json([]));
  render(<App />);

  const modes = within(screen.getByRole('group', { name: 'Search mode' })).getAllByRole('button');
  expect(modes.map((b) => b.textContent)).toEqual(['My files', 'Wiki dump']);
  expect(screen.getByRole('button', { name: 'My files' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByText('Ask questions about your files')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Wiki dump' }));
  expect(screen.getByRole('button', { name: 'Wiki dump' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByText('Ask questions about your files')).not.toBeInTheDocument();
  await act(async () => { await Promise.resolve(); });
});

async function upload(name) {
  const input = screen.getByLabelText('Choose text, Markdown, PDF, Word or MP3 files');
  fireEvent.change(input, { target: { files: [new File(['text'], name, { type: 'text/plain' })] } });
  fireEvent.submit(input.closest('form'));
  await screen.findByLabelText(`Select ${name}`);
}

const workspacePosts = () => global.fetch.mock.calls
  .filter(([url, o]) => url.endsWith('/api/v1/corpora') && o?.method === 'POST');

test('a first visit creates no workspace until a chat uploads its first file', async () => {
  window.localStorage.removeItem('graphrag-corpus-id');
  render(<App />);
  await act(async () => { await Promise.resolve(); });
  expect(workspacePosts()).toHaveLength(0);
  expect(screen.getByText(/Each chat has its own files/)).toBeInTheDocument();

  await upload('notes.txt');
  expect(workspacePosts()).toHaveLength(1);
  // A chat with files but no question yet is kept in the sidebar
  expect(chatRows()).toEqual(['Untitled chat']);
});

test("each chat has its own files: another chat's files are neither listed nor searched", async () => {
  window.confirm = jest.fn(() => true);
  routeSearches((prompt) => sseResponse([['done', { answer: `Answer to ${prompt}` }]]));
  await openMyFiles(); // the browser's existing workspace goes to the first chat
  ask('Who overtook Europe?');
  await screen.findByText('Answer to Who overtook Europe?');

  fireEvent.click(screen.getByRole('button', { name: '+ New chat' }));
  await waitFor(() => expect(screen.queryByLabelText('Select cinema.txt')).not.toBeInTheDocument());
  expect(screen.getByRole('button', { name: 'Search' })).toBeDisabled(); // nothing to search yet
  await upload('notes.txt');
  const second = newCorpus(1);
  expect(screen.queryByLabelText('Select cinema.txt')).not.toBeInTheDocument();
  ask('What do the notes say?');
  await screen.findByText('Answer to What do the notes say?');
  expect(searchBodies()[1]).toMatchObject({ corpusId: second, fileIds: [] });
  expect(searchBodies()[0]).toMatchObject({ corpusId: CORPUS });

  // Back in the first chat: only its own files
  fireEvent.click(screen.getByTitle('Who overtook Europe?'));
  await screen.findByLabelText('Select cinema.txt');
  expect(screen.queryByLabelText('Select notes.txt')).not.toBeInTheDocument();

  // Deleting a chat deletes its files from search too
  fireEvent.click(screen.getByRole('button', { name: 'Delete What do the notes say?' }));
  expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('and the files uploaded to it'));
  await waitFor(() => expect(global.fetch.mock.calls.some(([url, o]) => o?.method === 'DELETE'
    && url.includes(`/corpora/${second}/files/`))).toBe(true));
  expect(global.fetch.mock.calls.some(([url, o]) => o?.method === 'DELETE' && url.includes(CORPUS))).toBe(false);
});

test('the workspace older chats shared stays with them, and a new chat starts with its own', async () => {
  const { saveChat } = require('./chatStore');
  const oldChat = (id, title, updatedAt) => ({
    id, title, mode: 'files', corpusId: CORPUS, fileIds: [], createdAt: updatedAt, updatedAt, context: [],
    turns: [{ id: `${id}-t`, question: title, done: true, text: 'Old answer', agentSteps: [], sources: [],
      chunks: [], graphPaths: [], graphSeeds: [] }],
  });
  await saveChat(oldChat('old-1', 'Old question', 2));
  await saveChat(oldChat('old-2', 'Older question', 1));
  render(<App />);
  await screen.findByTitle('Old question');
  await act(async () => { await Promise.resolve(); });
  expect(screen.queryByLabelText('Select cinema.txt')).not.toBeInTheDocument(); // the new chat has none
  expect(window.localStorage.getItem('graphrag-corpus-id')).toBeNull();

  fireEvent.click(screen.getByTitle('Old question'));
  await screen.findByLabelText('Select cinema.txt');

  // Deleting one keeps the shared files: the other old chat still uses them
  window.confirm = jest.fn(() => true);
  fireEvent.click(screen.getByRole('button', { name: 'Delete Old question' }));
  expect(window.confirm).toHaveBeenCalledWith(expect.not.stringContaining('files'));
  expect(chatRows()).toEqual(['Older question']);
  await act(async () => { await Promise.resolve(); });
  expect(global.fetch.mock.calls.some(([, o]) => o?.method === 'DELETE')).toBe(false);
});

test('checked files scope the query; none checked searches all files', async () => {
  await openMyFiles();

  expect(screen.getByText('No files selected: questions will search all 2 files.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Preview' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled();

  fireEvent.click(screen.getByLabelText('Select tea.txt'));
  expect(screen.getByText('Questions will search the 1 selected file only.')).toBeInTheDocument();

  fireEvent.change(screen.getByPlaceholderText(/uploaded files/), { target: { value: 'Where was tea grown?' } });
  fireEvent.submit(screen.getByPlaceholderText(/uploaded files/).closest('form'));

  const searchCall = global.fetch.mock.calls.find(([url]) => url.endsWith('/api/v1/search'));
  expect(JSON.parse(searchCall[1].body)).toMatchObject({ corpusId: CORPUS, fileIds: [FILES[1].fileId] });
  await act(async () => { await Promise.resolve(); });
});

test('follow-ups send the earlier turns and keep earlier answers on screen; New topic clears the context', async () => {
  const { TextEncoder } = require('util');
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  const doneEvent = (data) => {
    const bytes = new TextEncoder().encode(`event:done\ndata:${JSON.stringify(data)}\n\n`);
    let sent = false;
    return Promise.resolve({
      ok: true,
      body: {
        getReader: () => ({
          read: () => Promise.resolve(sent ? { done: true } : ((sent = true), { done: false, value: bytes })),
        }),
      },
    });
  };
  const acmeSource = 's3://rag-datasets/uploads/acme.txt';
  const replies = [
    { answer: 'Acme Corp was founded by Ada.', entityKeys: ['acme corp'], standaloneQuery: 'Who founded Acme Corp?',
      sources: [acmeSource] },
    { answer: 'In 2001.', entityKeys: ['acme corp', 'beta inc'], standaloneQuery: 'When did Acme Corp acquire Beta Inc?' },
  ];
  const base = global.fetch;
  global.fetch = jest.fn((url, options) => (url.endsWith('/api/v1/search') ? doneEvent(replies.shift()) : base(url, options)));
  await openMyFiles();
  const input = screen.getByPlaceholderText(/uploaded files/);
  const ask = async (text) => {
    fireEvent.change(input, { target: { value: text } });
    fireEvent.submit(input.closest('form'));
    expect(input).toHaveValue(''); // the box clears once a question is sent
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Searching…' })).not.toBeInTheDocument());
  };
  const searchBodies = () => global.fetch.mock.calls
    .filter(([url]) => url.endsWith('/api/v1/search')).map(([, o]) => JSON.parse(o.body));

  await ask('Who founded Acme Corp?');
  await screen.findByText(/refer to your last question/);
  await ask('When did they acquire Beta Inc?');
  expect(searchBodies()[0].history).toEqual([]);
  expect(searchBodies()[1].history).toEqual([
    { question: 'Who founded Acme Corp?', answer: 'Acme Corp was founded by Ada.', entityKeys: ['acme corp'] },
  ]);
  expect(screen.getByText(/Searched as “When did Acme Corp acquire Beta Inc\?”/)).toBeInTheDocument();
  expect(screen.getByText(/refer to your last 2 questions/)).toBeInTheDocument();

  // Both turns stay on screen, in order, each with its own answer; the first keeps its source link
  const turns = screen.getAllByRole('article');
  expect(turns.map((t) => t.querySelector('.turn-question-text').textContent))
    .toEqual(['Who founded Acme Corp?', 'When did they acquire Beta Inc?']);
  expect(within(turns[0]).getByText('Acme Corp was founded by Ada.')).toBeInTheDocument();
  expect(within(turns[0]).getByRole('link', { name: /acme.*opens in a new tab/ })).toBeInTheDocument();
  expect(within(turns[1]).getByText('In 2001.')).toBeInTheDocument();
  expect(within(turns[1]).queryByRole('link')).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'New topic' }));
  expect(screen.queryByRole('button', { name: 'New topic' })).not.toBeInTheDocument();
  expect(screen.getByText(/New topic: the next question starts fresh/)).toBeInTheDocument();
  expect(screen.getAllByRole('article')).toHaveLength(2); // earlier answers are still there
});

test('Enter sends the question, Shift+Enter does not', async () => {
  await openMyFiles();
  const input = screen.getByPlaceholderText(/uploaded files/);
  const searches = () => global.fetch.mock.calls.filter(([url]) => url.endsWith('/api/v1/search')).length;

  fireEvent.change(input, { target: { value: 'Where was tea grown?' } });
  fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
  expect(searches()).toBe(0);
  fireEvent.keyDown(input, { key: 'Enter' });
  expect(searches()).toBe(1);
  await act(async () => { await Promise.resolve(); });
});

test('switching corpus starts a new chat; the earlier one stays in the sidebar and reopens in its corpus', async () => {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  await openMyFiles();
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: 's3://rag/x.txt',
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }]);
  expect(screen.getAllByRole('article')).toHaveLength(1);

  fireEvent.click(screen.getByRole('button', { name: 'Wiki dump' }));
  expect(screen.queryAllByRole('article')).toHaveLength(0);
  expect(screen.queryByRole('button', { name: 'New topic' })).not.toBeInTheDocument();

  const nav = screen.getByRole('navigation', { name: 'Chats' });
  fireEvent.click(within(nav).getByTitle('Who overtook Europe?'));
  expect(screen.getByRole('button', { name: 'My files' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getAllByRole('article')).toHaveLength(1);
});

// --- Several chats -----------------------------------------------------------------------

function sseResponse(events) {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  const text = events.map(([type, data]) => `event:${type}\ndata:${JSON.stringify(data)}\n\n`).join('');
  const reads = [{ done: false, value: new Uint8Array(Buffer.from(text)) }];
  return Promise.resolve({
    ok: true,
    body: { getReader: () => ({ read: () => Promise.resolve(reads.shift() || { done: true }) }) },
  });
}

// A search whose answer arrives only when the test calls finish()
function heldSearch() {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder;
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const reads = [];
  return {
    response: Promise.resolve({
      ok: true,
      body: { getReader: () => ({ read: () => gate.then(() => reads.shift() || { done: true }) }) },
    }),
    finish(data) {
      reads.push({ done: false, value: new Uint8Array(Buffer.from(`event:done\ndata:${JSON.stringify(data)}\n\n`)) });
      release();
    },
  };
}

// Search requests answered by reply(prompt, body); everything else by mockApi
function routeSearches(reply) {
  const base = global.fetch;
  global.fetch = jest.fn((url, options) => (url.endsWith('/api/v1/search')
    ? reply(JSON.parse(options.body).prompt, JSON.parse(options.body))
    : base(url, options)));
}

const searchBodies = () => global.fetch.mock.calls
  .filter(([url]) => url.endsWith('/api/v1/search')).map(([, o]) => JSON.parse(o.body));

function ask(text) {
  const input = screen.getByPlaceholderText(/uploaded files|wiki dump/);
  fireEvent.change(input, { target: { value: text } });
  fireEvent.submit(input.closest('form'));
}

const chatRows = () => [...screen.getByRole('navigation', { name: 'Chats' }).querySelectorAll('.chat-open')]
  .map((b) => b.title);

test('each chat keeps its own thread and follow-up context', async () => {
  const replies = {
    'Who founded Acme Corp?': { answer: 'Ada founded Acme Corp.', entityKeys: ['acme corp'] },
    'What is tea?': { answer: 'Tea is a drink.', entityKeys: ['tea'] },
    'When did they buy Beta Inc?': { answer: 'In 2001.', entityKeys: ['acme corp', 'beta inc'],
      standaloneQuery: 'When did Acme Corp buy Beta Inc?' },
  };
  routeSearches((prompt) => sseResponse([['done', replies[prompt]]]));
  await openMyFiles();

  ask('Who founded Acme Corp?');
  await screen.findByText('Ada founded Acme Corp.');
  fireEvent.click(screen.getByRole('button', { name: '+ New chat' }));
  expect(screen.queryAllByRole('article')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Wiki dump' })); // the new chat has no files
  ask('What is tea?');
  await screen.findByText('Tea is a drink.');
  expect(chatRows()).toEqual(['What is tea?', 'Who founded Acme Corp?']); // newest first

  // Back to the first chat: only its thread, and follow-ups use only its turns
  fireEvent.click(screen.getByTitle('Who founded Acme Corp?'));
  expect(screen.getAllByRole('article')).toHaveLength(1);
  expect(screen.queryByText('Tea is a drink.')).not.toBeInTheDocument();
  await screen.findByLabelText('Select cinema.txt'); // its own files load again
  ask('When did they buy Beta Inc?');
  await screen.findByText('In 2001.');
  expect(searchBodies()[2].history).toEqual([
    { question: 'Who founded Acme Corp?', answer: 'Ada founded Acme Corp.', entityKeys: ['acme corp'] },
  ]);
  expect(chatRows()).toEqual(['Who founded Acme Corp?', 'What is tea?']); // just used: back on top
});

test('chats are saved in the browser and come back after a reload; an unfinished answer says it was interrupted', async () => {
  const held = heldSearch();
  routeSearches((prompt) => (prompt === 'Who founded Acme Corp?'
    ? sseResponse([['done', { answer: 'Ada founded Acme Corp.', entityKeys: ['acme corp'] }]])
    : held.response));
  const first = render(<App />);
  await screen.findByLabelText('Select cinema.txt');
  ask('Who founded Acme Corp?');
  await screen.findByText('Ada founded Acme Corp.');
  ask('When did they buy Beta Inc?');
  await screen.findByRole('button', { name: 'Searching…' });
  first.unmount(); // the page is closed mid-answer

  render(<App />);
  const row = await within(screen.getByRole('navigation', { name: 'Chats' })).findByTitle('Who founded Acme Corp?');
  expect(screen.queryAllByRole('article')).toHaveLength(0); // a reload opens a new chat
  fireEvent.click(row);
  const turns = screen.getAllByRole('article');
  expect(turns).toHaveLength(2);
  expect(within(turns[0]).getByText('Ada founded Acme Corp.')).toBeInTheDocument();
  expect(within(turns[1]).getByRole('alert')).toHaveTextContent('This answer was interrupted');
});

test('one search at a time: another chat waits, and the running answer keeps writing into its own chat', async () => {
  const held = heldSearch();
  routeSearches((prompt) => (prompt === 'Who founded Acme Corp?' ? held.response
    : sseResponse([['done', { answer: 'Tea is a drink.' }]])));
  await openMyFiles();

  ask('Who founded Acme Corp?');
  await screen.findByRole('button', { name: 'Searching…' });
  fireEvent.click(screen.getByRole('button', { name: '+ New chat' }));
  fireEvent.click(screen.getByRole('button', { name: 'Wiki dump' })); // the new chat has no files
  const nav = screen.getByRole('navigation', { name: 'Chats' });
  expect(within(nav).getByLabelText('Answering')).toBeInTheDocument();

  // The new chat can type, but not send, until the other answer is done
  const input = screen.getByPlaceholderText(/wiki dump/);
  fireEvent.change(input, { target: { value: 'What is tea?' } });
  expect(screen.getByRole('button', { name: 'Waiting…' })).toBeDisabled();
  expect(screen.getByText(/“Who founded Acme Corp\?” is still being answered/)).toBeInTheDocument();
  fireEvent.keyDown(input, { key: 'Enter' });
  expect(searchBodies()).toHaveLength(1);

  await act(async () => { held.finish({ answer: 'Ada founded Acme Corp.' }); });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Search' })).toBeEnabled());
  expect(within(nav).queryByLabelText('Answering')).not.toBeInTheDocument();
  ask('What is tea?');
  await screen.findByText('Tea is a drink.');

  // The first chat got its answer while it wasn't open
  fireEvent.click(screen.getByTitle('Who founded Acme Corp?'));
  expect(screen.getByText('Ada founded Acme Corp.')).toBeInTheDocument();
});

test('sources and evidence from a file deleted later say it is no longer available', async () => {
  window.confirm = jest.fn(() => true);
  const uri = `s3://rag-datasets/uploads/${CORPUS}/${CINEMA}/cinema.txt`;
  routeSearches(() => sseResponse([['done', {
    answer: 'Hollywood overtook Europe.', sources: [uri],
    chunks: [{ chunkId: `${CINEMA}:0`, content: 'Hollywood overtook Europe.', source: uri,
      milvusScore: 1, pagerankScore: 1, hybridScore: 1 }],
  }]]));
  await openMyFiles();
  ask('Who overtook Europe?');
  await screen.findByText('Evidence (1 chunks)');
  const turn = screen.getByRole('article');
  expect(within(turn).getByRole('link', { name: /cinema\.txt/ })).toBeInTheDocument();
  expect(within(turn).getByRole('button', { name: 'View full source' })).toBeInTheDocument();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete (1)' }));
  await waitFor(() => expect(screen.queryByLabelText('Select cinema.txt')).not.toBeInTheDocument());

  expect(within(turn).queryByRole('link', { name: /cinema\.txt/ })).not.toBeInTheDocument();
  expect(within(turn).getByText('No longer available: the file was deleted')).toBeInTheDocument();
  expect(within(turn).getByText('File deleted')).toBeInTheDocument();
  expect(within(turn).getByText('Hollywood overtook Europe.', { selector: '.evidence-snippet' })).toBeInTheDocument();
  expect(within(turn).queryByRole('button', { name: 'View full source' })).not.toBeInTheDocument();
  expect(within(turn).queryByRole('button', { name: 'Show in map' })).not.toBeInTheDocument();
});

test('chats can be renamed and deleted', async () => {
  window.confirm = jest.fn(() => true);
  routeSearches(() => sseResponse([['done', { answer: 'Ada founded Acme Corp.' }]]));
  await openMyFiles();
  ask('Who founded Acme Corp?');
  await screen.findByText('Ada founded Acme Corp.');

  fireEvent.click(screen.getByRole('button', { name: 'Rename Who founded Acme Corp?' }));
  const name = screen.getByLabelText('Chat name');
  fireEvent.change(name, { target: { value: 'Acme notes' } });
  fireEvent.keyDown(name, { key: 'Enter' });
  expect(chatRows()).toEqual(['Acme notes']);

  fireEvent.click(screen.getByRole('button', { name: 'Delete Acme notes' }));
  expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Acme notes'));
  expect(chatRows()).toEqual([]);
  expect(screen.queryAllByRole('article')).toHaveLength(0); // the open chat was deleted: a new one opens
});

test('select all toggles every file', async () => {
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select all files'));
  expect(screen.getByLabelText('Select cinema.txt')).toBeChecked();
  expect(screen.getByLabelText('Select tea.txt')).toBeChecked();
  expect(screen.getByRole('button', { name: 'Delete (2)' })).toBeEnabled();

  fireEvent.click(screen.getByLabelText('Select all files'));
  expect(screen.getByLabelText('Select cinema.txt')).not.toBeChecked();
});

test('previews the full text of selected files', async () => {
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Preview (1)' }));

  expect(await screen.findByText('Full text of cinema.txt')).toBeInTheDocument();
  expect(screen.queryByText('Full text of tea.txt')).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Close' }));
  expect(screen.queryByText('Full text of cinema.txt')).not.toBeInTheDocument();
});

test('deletes selected files after confirmation', async () => {
  window.confirm = jest.fn(() => true);
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete (1)' }));

  expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('cinema.txt'));
  await waitFor(() => expect(screen.queryByLabelText('Select cinema.txt')).not.toBeInTheDocument());
  expect(screen.getByLabelText('Select tea.txt')).toBeInTheDocument();
  expect(screen.getByText('Deleted 1 file.')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining(`/files/${FILES[0].fileId}`), { method: 'DELETE' });
});

test('shows a transcription notice while an MP3 is uploading', async () => {
  await openMyFiles();
  let finishUpload;
  global.fetch.mockImplementationOnce(() => new Promise((resolve) => { finishUpload = resolve; }));

  const input = screen.getByLabelText('Choose text, Markdown, PDF, Word or MP3 files');
  fireEvent.change(input, { target: { files: [new File(['ID3'], 'lecture.mp3', { type: 'audio/mpeg' })] } });
  fireEvent.submit(input.closest('form'));

  expect(screen.getByText(/Transcribing audio before indexing/)).toBeInTheDocument();
  await act(async () => {
    finishUpload({ ok: true, json: () => Promise.resolve({ files: [
      { success: true, fileId: '33333333-3333-4333-8333-333333333333', filename: 'lecture.mp3', chunkCount: 2 },
    ] }) });
  });
  expect(await screen.findByLabelText('Select lecture.mp3')).toBeInTheDocument();
  expect(screen.getByText('1 file indexed and ready.')).toBeInTheDocument();
});

test('shows evidence when the done event arrives split across several reads', async () => {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  await openMyFiles();
  const done = JSON.stringify({
    answer: 'Both cite capital.', sources: [], confidence: 0.95,
    chunks: [{ chunkId: 'c1', content: 'x'.repeat(20000), source: 's3://rag/a.pdf',
      milvusScore: 0.9, pagerankScore: 0.1, hybridScore: 0.5 }],
  });
  const stream = `event:chunk\ndata:{"content":"Both "}\n\nevent:done\ndata:${done}\n\n`;
  // Cut so "event:done" lands in one read and its data line in later ones
  const cut = stream.indexOf('data:', stream.indexOf('event:done')) + 10;
  const parts = [stream.slice(0, cut), stream.slice(cut, cut + 8000), stream.slice(cut + 8000)];
  const reads = parts.map((p) => ({ done: false, value: new Uint8Array(Buffer.from(p)) }));
  global.fetch.mockImplementationOnce(() => Promise.resolve({
    ok: true,
    body: { getReader: () => ({ read: () => Promise.resolve(reads.shift() || { done: true }) }) },
  }));

  fireEvent.change(screen.getByPlaceholderText(/uploaded files/), { target: { value: 'Which reasons?' } });
  fireEvent.submit(screen.getByPlaceholderText(/uploaded files/).closest('form'));

  expect(await screen.findByText('Evidence (1 chunks)')).toBeInTheDocument();
});

test('does not delete when the confirmation is cancelled', async () => {
  window.confirm = jest.fn(() => false);
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete (1)' }));

  expect(screen.getByLabelText('Select cinema.txt')).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalledWith(expect.anything(), { method: 'DELETE' });
});

test('corpus map shows entities, relationships and clusters for the selected files', async () => {
  await openMyFiles();
  fireEvent.click(screen.getByLabelText('Select tea.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Map (1)' }));

  expect(await screen.findByText(/3 entities · 2 relationships · 2 clusters/)).toBeInTheDocument();
  const mapCall = global.fetch.mock.calls.find(([url]) => url.includes('/map'));
  expect(mapCall[0]).toContain(`/corpora/${CORPUS}/map?fileIds=${TEA}`);

  // Selecting an entity lists its relationships with the evidence sentence
  fireEvent.click(screen.getByRole('button', { name: /^Hollywood 2$/ }));
  expect(screen.getByRole('heading', { name: 'Hollywood' })).toBeInTheDocument();
  expect(screen.getByText(/Hollywood had better special effects./, { selector: 'blockquote' })).toBeInTheDocument();
  expect(screen.getByText(/across 2 files/)).toBeInTheDocument();

  // Concept filter hides names
  fireEvent.click(screen.getByRole('button', { name: 'Concepts' }));
  expect(screen.queryByRole('button', { name: /^Europe 1$/ })).not.toBeInTheDocument();

  // Clusters tab: shared theme flagged, chunks expandable
  fireEvent.click(screen.getByRole('tab', { name: 'Document clusters' }));
  expect(screen.getByText('Theme 1: Hollywood, special effects')).toBeInTheDocument();
  expect(screen.getByText('Shared by 2 files')).toBeInTheDocument();
  fireEvent.click(screen.getAllByRole('button', { name: 'Show 2 chunks' })[0]);
  expect(screen.getByText('Effects drew crowds.')).toBeInTheDocument();

  // No Rebuild button: a changed selection points to the Map button instead
  expect(screen.queryByRole('button', { name: /Rebuild/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  expect(screen.getByText(/File selection changed. Click Map to map the 2 selected files/)).toBeInTheDocument();

  // Clicking Map again builds the map for the new selection, like Preview
  fireEvent.click(screen.getByRole('button', { name: 'Map (2)' }));
  const mapCalls = global.fetch.mock.calls.filter(([url]) => url.includes('/map'));
  expect(mapCalls).toHaveLength(2);
  expect(mapCalls[1][0]).toContain(`/map?fileIds=${CINEMA},${TEA}`);
  await waitFor(() => expect(screen.queryByText(/File selection changed/)).not.toBeInTheDocument());
});

test('clicking Map with the same selection fetches the map again (picks up new facts)', async () => {
  await openMyFiles();
  fireEvent.click(screen.getByRole('button', { name: 'Map' }));
  await screen.findByText(/3 entities/);
  fireEvent.click(screen.getByRole('button', { name: 'Map' }));
  expect(screen.getByText('Updating the map…')).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByText('Updating the map…')).not.toBeInTheDocument());
  expect(global.fetch.mock.calls.filter(([url]) => url.includes('/map'))).toHaveLength(2);
});

test('corpus map shows the server error', async () => {
  await openMyFiles();
  global.fetch.mockImplementationOnce(() => json({ message: 'No uploaded files to map' }, false, 400));
  fireEvent.click(screen.getByRole('button', { name: 'Map' }));
  expect(await screen.findByText('No uploaded files to map')).toBeInTheDocument();
});

test('corpus map shows extracted facts with direction, evidence and extraction progress', async () => {
  const kgMap = {
    ...MAP,
    mode: 'knowledge_graph',
    progress: { enabled: true, done: 2, failed: 0, pending: 1, total: 3 },
    relationships: [
      { source: 'europe', target: 'hollywood', weight: 1, fileIds: [CINEMA],
        examples: [{ chunkId: `${CINEMA}:0`, filename: 'cinema.txt', sentence: 'Hollywood overtook Europe.',
          from: 'hollywood', relation: 'overtook', to: 'europe' }] },
    ],
  };
  await openMyFiles();
  global.fetch.mockImplementationOnce(() => json(kgMap));
  fireEvent.click(screen.getByRole('button', { name: 'Map' }));

  expect(await screen.findByText(/Extracting facts with the local model/)).toHaveTextContent('2 of 3 chunks read');
  expect(screen.getByText(/Fact stated in your files/)).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /^Europe 1$/ }));
  expect(screen.getByRole('heading', { name: 'Facts' })).toBeInTheDocument();
  const fact = screen.getByText('overtook', { selector: '.fact-relation' }).closest('p');
  expect(fact).toHaveTextContent('Hollywood overtook Europe');
  expect(screen.getByText(/Hollywood overtook Europe./, { selector: 'blockquote' })).toBeInTheDocument();
});

// Runs a search whose answer cites the given evidence chunks
async function searchWithEvidence(evidence, extra = {}) {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  const done = JSON.stringify({ answer: 'Hollywood overtook Europe.', sources: [], confidence: 0.9, chunks: evidence, ...extra });
  const reads = [{ done: false, value: new Uint8Array(Buffer.from(`event:done\ndata:${done}\n\n`)) }];
  global.fetch.mockImplementationOnce(() => Promise.resolve({
    ok: true,
    body: { getReader: () => ({ read: () => Promise.resolve(reads.shift() || { done: true }) }) },
  }));
  fireEvent.change(screen.getByPlaceholderText(/uploaded files/), { target: { value: 'Who overtook Europe?' } });
  fireEvent.submit(screen.getByPlaceholderText(/uploaded files/).closest('form'));
  await screen.findByText(`Evidence (${evidence.length} chunks)`);
}

test('hovering an evidence chunk highlights its entities in the corpus map; Show in map pins it', async () => {
  const chunk = (chunkId, content) => ({ chunkId, content, source: 's3://rag/x.txt',
    milvusScore: 0.9, pagerankScore: 0.1, hybridScore: 0.5 });
  await openMyFiles();
  await searchWithEvidence([chunk(`${CINEMA}:0`, 'Hollywood overtook Europe.'), chunk(`${TEA}:7`, 'Not mapped.')]);

  // Map closed: "Show in map" opens it, scrolls to it and keeps the highlight
  const [first, second] = screen.getAllByRole('listitem').filter((li) => li.classList.contains('evidence-item'));
  fireEvent.click(within(first).getByRole('button', { name: 'Show in map' }));
  const banner = await screen.findByText(/Evidence:/);
  expect(banner.closest('p')).toHaveTextContent('Evidence: cinema.txt · chunk 1. 2 entities highlighted.');
  expect(scrolledTo().some((el) => el.classList.contains('corpus-map-anchor'))).toBe(true);
  const ringed = () => [...document.querySelectorAll('.node.in-evidence')].map((n) => n.getAttribute('aria-label').split(',')[0]);
  expect(ringed()).toEqual(['Hollywood', 'Europe']);
  expect(document.querySelector('.node.dim').getAttribute('aria-label')).toMatch(/^special effects/);
  expect(within(first).getByRole('button', { name: 'Stop highlighting in map' })).toHaveAttribute('aria-pressed', 'true');

  // Hovering another chunk takes over while hovered; one outside the map says so
  fireEvent.mouseEnter(second);
  expect(screen.getByText(/isn't in this map/)).toBeInTheDocument();
  expect(ringed()).toEqual([]);
  fireEvent.mouseLeave(second);
  expect(screen.getByText(/Evidence:/).closest('p')).toHaveTextContent('cinema.txt · chunk 1');

  // Clear removes the pinned highlight
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(screen.queryByText(/Evidence:/)).not.toBeInTheDocument();
  expect(ringed()).toEqual([]);

  // Plain hover with the map open
  fireEvent.mouseEnter(first);
  expect(ringed()).toEqual(['Hollywood', 'Europe']);
  fireEvent.mouseLeave(first);
  expect(ringed()).toEqual([]);
});

test('shows the knowledge-graph reasoning path, marks graph-found evidence, and highlights a path in the map', async () => {
  Element.prototype.scrollIntoView = jest.fn(); // jsdom lacks it
  const fact = (source, sourceLabel, relation, target, targetLabel, chunkId, filename) =>
    ({ source, sourceLabel, relation, target, targetLabel, chunkId, fileId: chunkId.split(':')[0], filename, evidence: 'x' });
  const graphPaths = [
    { score: 0.66, facts: [fact('hollywood', 'Hollywood', 'overtook', 'europe', 'Europe', `${CINEMA}:0`, 'cinema.txt')] },
    { score: 0.44, facts: [
      fact('hollywood', 'Hollywood', 'overtook', 'europe', 'Europe', `${CINEMA}:0`, 'cinema.txt'),
      fact('united states government', 'United States government', 'supported', 'hollywood', 'Hollywood', `${TEA}:0`, 'tea.txt'),
    ] },
  ];
  await openMyFiles();
  await searchWithEvidence([
    { chunkId: `${CINEMA}:0`, content: 'Hollywood overtook Europe.', source: `s3://rag/uploads/${CORPUS}/${CINEMA}/cinema.pdf.txt`, milvusScore: 0.4,
      pagerankScore: 0.1, graphScore: 1, foundByGraph: true, hybridScore: 0.6 },
    { chunkId: `${TEA}:7`, content: 'Other.', source: 's3://rag/y.txt', milvusScore: 1, pagerankScore: 0.2,
      graphScore: 0, foundByGraph: false, hybridScore: 0.5 },
  ], { graphPaths, graphSeeds: ['Europe'] });

  // Reasoning section: start entity, each path written as facts, 2-hop path joined by "then"
  expect(screen.getByRole('heading', { name: 'Reasoning path' })).toBeInTheDocument();
  expect(screen.getByText(/followed from Europe/)).toBeInTheDocument();
  const paths = document.querySelectorAll('.graph-path');
  expect(paths).toHaveLength(2);
  expect(paths[1]).toHaveTextContent('Hollywood overtook Europe · cinema.txtthenUnited States government supported Hollywood · tea.txt');

  // Evidence: graph-found chunk is labelled and shows its graph score; the other has neither
  const [first, second] = document.querySelectorAll('.evidence-item');
  expect(within(first).getByText('Found via knowledge graph')).toBeInTheDocument();
  expect(within(first).getByText('Graph 100%')).toBeInTheDocument();
  expect(first.querySelector('.evidence-source')).toHaveTextContent(/^cinema\.pdf$/); // stored name shown as uploaded
  expect(within(second).queryByText(/Found via knowledge graph|Graph \d/)).not.toBeInTheDocument();

  // Show the 2-hop path in the map: its entities ringed, the missing one reported
  fireEvent.click(within(paths[1]).getByRole('button', { name: 'Show in map' }));
  const banner = await screen.findByText(/Reasoning path:/);
  expect(banner.closest('p')).toHaveTextContent(
    'Reasoning path: 2 facts linking Hollywood, Europe, United States government. 1 entity not in this map');
  const ringed = () => [...document.querySelectorAll('.node.in-evidence')].map((n) => n.getAttribute('aria-label').split(',')[0]);
  expect(ringed()).toEqual(['Hollywood', 'Europe']);
  expect(screen.getByRole('button', { name: 'Back to reasoning' })).toBeInTheDocument();
  expect(document.querySelector('.link.active')).not.toBeNull();

  // Hovering the other path switches the highlight while hovered
  fireEvent.mouseEnter(paths[0]);
  expect(screen.getByText(/Reasoning path:/).closest('p')).toHaveTextContent('1 fact linking Hollywood, Europe.');
  fireEvent.mouseLeave(paths[0]);
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(ringed()).toEqual([]);
});

test('shows the facts two files share for a file-similarity answer', async () => {
  Element.prototype.scrollIntoView = jest.fn();
  const fact = (source, sourceLabel, relation, target, targetLabel, chunkId, filename) =>
    ({ source, sourceLabel, relation, target, targetLabel, chunkId, fileId: chunkId.split(':')[0], filename, evidence: 'x' });
  const graphPaths = [{ score: 0.75, kind: 'shared', facts: [
    fact('hollywood', 'Hollywood', 'overtook', 'europe', 'Europe', `${CINEMA}:0`, 'cinema.txt'),
    fact('hollywood', 'Hollywood', 'surpassed', 'europe', 'Europe', `${TEA}:0`, 'tea.txt'),
  ] }];
  await openMyFiles();
  await searchWithEvidence([
    { chunkId: `${CINEMA}:0`, content: 'Hollywood overtook Europe.', source: `s3://rag/uploads/${CORPUS}/${CINEMA}/cinema.txt`,
      milvusScore: 1, pagerankScore: 0, graphScore: 0, foundByGraph: false, hybridScore: 0.6 },
  ], { graphPaths, graphSeeds: [] });

  expect(screen.getByRole('heading', { name: 'Shared facts' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Reasoning path' })).not.toBeInTheDocument();
  const [path] = document.querySelectorAll('.graph-path');
  expect(path).toHaveTextContent('Hollywood overtook Europe · cinema.txtsame asHollywood surpassed Europe · tea.txt');

  fireEvent.click(within(path).getByRole('button', { name: 'Show in map' }));
  const banner = await screen.findByText(/Shared facts:/);
  expect(banner.closest('p')).toHaveTextContent('Shared facts: 2 facts linking Hollywood, Europe.');
  expect(screen.getByRole('button', { name: 'Back to shared facts' })).toBeInTheDocument();
});

test('shows the answer as formatted Markdown, answer first, with the checks collapsed', async () => {
  await openMyFiles();
  const answer = [
    '### Checked evidence',
    '- **plenty of capital**: SUPPORTED. "Hollywood had the money." [cinema.txt]',
    '- **serious issues**: NOT SUPPORTED. "Swedish films were serious." [cinema.txt]',
    '',
    '### Answer',
    'The reasons are **plenty of capital** and **well-written narratives** [cinema.txt].',
  ].join('\n');
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: 's3://rag/x.txt',
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }], { answer });

  const section = document.querySelector('.answer-section');
  const text = section.querySelector('.answer-text');
  expect(text).toHaveTextContent('The reasons are plenty of capital and well-written narratives cinema.txt');
  expect([...text.querySelectorAll('strong')].map((s) => s.textContent)).toEqual(['plenty of capital', 'well-written narratives']);
  expect(section.textContent).not.toContain('**');
  const checks = section.querySelector('details.answer-checks');
  expect(checks.open).toBe(false);
  expect(within(checks).getByText('How the answer was checked')).toBeInTheDocument();
  expect(checks.querySelectorAll('li')).toHaveLength(2);
});

test('a new question scrolls into view, with its answer written below it, within the chat only', async () => {
  await openMyFiles();
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: 's3://rag/x.txt',
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }]);
  const call = scrollInChat.mock.calls.filter(([, el]) => el?.classList.contains('thread-item')).at(-1);
  expect(call[0]).toHaveClass('chat-scroll'); // never the page around it
  const latest = call[1];
  expect(latest.querySelector('.turn-question-text')).toHaveTextContent('Who overtook Europe?');
  expect(latest.querySelector('.answer-section')).not.toBeNull();
});

test('Map and Preview scroll to what they open; Map scrolls back to an open map', async () => {
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Preview (1)' }));
  expect(scrolledTo().at(-1)).toHaveClass('file-preview');

  fireEvent.click(screen.getByRole('button', { name: 'Map (1)' }));
  expect(scrolledTo().at(-1).querySelector('.corpus-map')).not.toBeNull();
  await screen.findByText(/3 entities/);

  // Already open: the button stays enabled, scrolls back to the map and updates it
  scrollInChat.mockClear();
  const mapButton = screen.getByRole('button', { name: 'Map (1)' });
  expect(mapButton).toBeEnabled();
  fireEvent.click(mapButton);
  expect(scrolledTo().at(-1).querySelector('.corpus-map')).not.toBeNull();
  await waitFor(() => expect(screen.queryByText('Updating the map…')).not.toBeInTheDocument());
});

test('Sources are links that open the full text in a new tab', async () => {
  await openMyFiles();
  const uri = `s3://rag-datasets/uploads/${CORPUS}/${CINEMA}/cinema.pdf.txt`;
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: uri,
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }], { sources: [uri] });

  const link = screen.getByRole('link', { name: /cinema\.pdf.*opens in a new tab/ });
  expect(link).toHaveAttribute('href', `/?source=${encodeURIComponent(uri)}`);
  expect(link).toHaveAttribute('target', '_blank');
  expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  expect(link).toHaveTextContent(`cinema.pdfrag-datasets/uploads/${CORPUS}/${CINEMA}/cinema.pdf.txt`);
});

test('multiple choice: direct answer summary first, then every option with its verdict', async () => {
  const replies = require('./__fixtures__/modelReplies.json');
  await openMyFiles();
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: 's3://rag/x.txt',
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }], { answer: replies.bothFiles });

  const section = document.querySelector('.answer-section');
  const summary = section.querySelector('.answer-summary');
  expect(summary.querySelector('strong')).toHaveTextContent('Answer: 4 (well-written narratives)');
  expect(screen.getByRole('heading', { name: /Options evaluated/ })).toHaveTextContent('1 of 5 supported');
  const items = [...section.querySelectorAll('.option-item')];
  expect(items.map((li) => li.querySelector('.option-verdict').textContent))
    .toEqual(['✗Not supported', '✗Not supported', '✗Not supported', '✓Supported', '✗Not supported']);
  expect(items[3]).toHaveClass('is-supported');
  expect(items[3].querySelector('.option-label')).toHaveTextContent('well-written narratives');
  expect([...items[3].querySelectorAll('.option-evidence .md-cite')].map((c) => c.textContent)).toEqual(['sample.docx', 'sample2.pdf']);
  // The summary comes before the breakdown, and no raw Markdown or collapsed checks remain
  expect(summary.compareDocumentPosition(items[0]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(section.textContent).not.toContain('**');
  expect(section.querySelector('details.answer-checks')).toBeNull();
});
