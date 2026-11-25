import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import { resetMemoryStore } from './chatStore';

// Records the text of every answer drawn, so the test can tell which turns were redrawn
jest.mock('./Markdown', () => {
  const actual = jest.requireActual('./Markdown');
  const drawn = [];
  const Recorded = (props) => {
    drawn.push(props.text);
    return actual.default(props);
  };
  return { __esModule: true, ...actual, default: Recorded, drawn };
});
const { drawn } = require('./Markdown');

const CORPUS = '00000000-0000-4000-8000-000000000001';
const FILES = [{ fileId: '11111111-1111-4111-8111-111111111111', filename: 'cinema.txt' }];

function sseResponse(events) {
  const text = events.map(([type, data]) => `event:${type}\ndata:${JSON.stringify(data)}\n\n`).join('');
  const reads = [{ done: false, value: new Uint8Array(Buffer.from(text)) }];
  return Promise.resolve({
    ok: true,
    body: { getReader: () => ({ read: () => Promise.resolve(reads.shift() || { done: true }) }) },
  });
}

beforeEach(() => {
  global.TextDecoder = global.TextDecoder || require('util').TextDecoder; // jsdom lacks it
  window.localStorage.setItem('graphrag-corpus-id', CORPUS);
  resetMemoryStore();
  Element.prototype.scrollIntoView = jest.fn();
  global.fetch = jest.fn((url, options = {}) => {
    if (url.endsWith('/api/v1/search')) {
      const { prompt } = JSON.parse(options.body);
      if (prompt === 'First?') return sseResponse([['done', { answer: 'First answer.' }]]);
      // The second answer streams in piece by piece
      return sseResponse([
        ['chunk', { content: 'Second ' }], ['chunk', { content: 'answer ' }], ['chunk', { content: 'here.' }],
        ['done', { answer: 'Second answer here.' }],
      ]);
    }
    if (url.endsWith(`/corpora/${CORPUS}/files`)) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(FILES) });
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  });
});

test('a streaming answer and typing in the question box do not redraw earlier answers', async () => {
  render(<App />);
  await screen.findByLabelText('Select cinema.txt');
  const input = screen.getByPlaceholderText(/uploaded files/);
  const firstDrawn = () => drawn.filter((t) => t === 'First answer.').length;

  fireEvent.change(input, { target: { value: 'First?' } });
  fireEvent.submit(input.closest('form'));
  await screen.findByText('First answer.');
  const before = firstDrawn();

  fireEvent.change(input, { target: { value: 'Second?' } });
  fireEvent.submit(input.closest('form'));
  await screen.findByText('Second answer here.');
  fireEvent.change(input, { target: { value: 'Typing the next question…' } });

  expect(firstDrawn()).toBe(before);
  expect(drawn.filter((t) => t.startsWith('Second')).length).toBeGreaterThan(0);
});
