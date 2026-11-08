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

test('does not delete when the confirmation is cancelled', async () => {
  window.confirm = jest.fn(() => false);
  await openMyFiles();

  fireEvent.click(screen.getByLabelText('Select cinema.txt'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete (1)' }));

  expect(screen.getByLabelText('Select cinema.txt')).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalledWith(expect.anything(), { method: 'DELETE' });
});
