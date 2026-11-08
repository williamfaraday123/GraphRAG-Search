import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

const CORPUS = '00000000-0000-4000-8000-000000000001';
const FILES = [
  { fileId: '11111111-1111-4111-8111-111111111111', filename: 'cinema.txt' },
  { fileId: '22222222-2222-4222-8222-222222222222', filename: 'tea.txt' },
];

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
});

test('switches between wiki and uploaded-file search modes', async () => {
  global.fetch = jest.fn(() => json([]));
  render(<App />);

  expect(screen.getByRole('button', { name: 'Wiki dump' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'My files' }));
  expect(screen.getByText('Ask questions about your files')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'My files' })).toHaveAttribute('aria-pressed', 'true');
  await act(async () => { await Promise.resolve(); });
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
