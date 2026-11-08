import { act, fireEvent, render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  window.localStorage.setItem('graphrag-corpus-id', '00000000-0000-4000-8000-000000000001');
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve([]),
  });
});

test('switches between wiki and uploaded-file search modes', async () => {
  render(<App />);

  expect(screen.getByRole('button', { name: 'Wiki dump' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'My files' }));
  expect(screen.getByText('Ask questions about your files')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'My files' })).toHaveAttribute('aria-pressed', 'true');
  await act(async () => { await Promise.resolve(); });
});
