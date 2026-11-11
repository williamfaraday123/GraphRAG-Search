import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from './App';

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

// Routes fetch calls by method + URL so each test can inspect what the UI sent.
function mockApi({ files = FILES } = {}) {
  global.fetch = jest.fn((url, options = {}) => {
    const method = options.method || 'GET';
    if (url.endsWith(`/corpora/${CORPUS}/files`) && method === 'GET') return json(files);
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
  mockApi();
  Element.prototype.scrollIntoView = jest.fn(); // jsdom lacks it; Map/Preview scroll to what they open
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

test('a first visit creates an upload workspace on load, once', async () => {
  window.localStorage.removeItem('graphrag-corpus-id');
  const created = '00000000-0000-4000-8000-000000000099';
  global.fetch = jest.fn((url, options = {}) => {
    if (url.endsWith('/api/v1/corpora') && options.method === 'POST') return json({ corpusId: created });
    return json([]);
  });
  render(<App />);

  await waitFor(() => expect(window.localStorage.getItem('graphrag-corpus-id')).toBe(created));
  const posts = global.fetch.mock.calls.filter(([url, o]) => url.endsWith('/api/v1/corpora') && o?.method === 'POST');
  expect(posts).toHaveLength(1);
  await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining(`/corpora/${created}/files`)));
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
  Element.prototype.scrollIntoView = jest.fn(); // jsdom lacks it
  const chunk = (chunkId, content) => ({ chunkId, content, source: 's3://rag/x.txt',
    milvusScore: 0.9, pagerankScore: 0.1, hybridScore: 0.5 });
  await openMyFiles();
  await searchWithEvidence([chunk(`${CINEMA}:0`, 'Hollywood overtook Europe.'), chunk(`${TEA}:7`, 'Not mapped.')]);

  // Map closed: "Show in map" opens it, scrolls to it and keeps the highlight
  const [first, second] = screen.getAllByRole('listitem').filter((li) => li.classList.contains('evidence-item'));
  fireEvent.click(within(first).getByRole('button', { name: 'Show in map' }));
  const banner = await screen.findByText(/Evidence:/);
  expect(banner.closest('p')).toHaveTextContent('Evidence: cinema.txt · chunk 1. 2 entities highlighted.');
  expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
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

test('scrolls to the answer once it is complete', async () => {
  const scrolled = [];
  Element.prototype.scrollIntoView = jest.fn(function scroll() { scrolled.push(this); });
  await openMyFiles();
  await searchWithEvidence([{ chunkId: `${CINEMA}:0`, content: 'x', source: 's3://rag/x.txt',
    milvusScore: 1, pagerankScore: 1, hybridScore: 1 }]);
  await waitFor(() => expect(scrolled.at(-1)).toHaveClass('answer-section'));
});

test('Map and Preview scroll to what they open; Map scrolls back to an open map', async () => {
  const scrolled = [];
  Element.prototype.scrollIntoView = jest.fn(function scroll() { scrolled.push(this); });
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Preview (1)' }));
  expect(scrolled.at(-1)).toHaveClass('file-preview');

  fireEvent.click(screen.getByRole('button', { name: 'Map (1)' }));
  expect(scrolled.at(-1).querySelector('.corpus-map')).not.toBeNull();
  await screen.findByText(/3 entities/);

  // Already open: the button stays enabled, scrolls back to the map and updates it
  scrolled.length = 0;
  const mapButton = screen.getByRole('button', { name: 'Map (1)' });
  expect(mapButton).toBeEnabled();
  fireEvent.click(mapButton);
  expect(scrolled.at(-1).querySelector('.corpus-map')).not.toBeNull();
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
