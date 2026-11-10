import { render, screen } from '@testing-library/react';
import SourceView from './SourceView';

const URI = 's3://rag-datasets/uploads/c1/f1/report.pdf.txt';

test('shows the full text of a source with its file name, path and word count', async () => {
  global.fetch = jest.fn(() => Promise.resolve({
    ok: true, json: () => Promise.resolve({ found: true, content: 'Hollywood overtook Europe.\n\nSecond paragraph.' }),
  }));
  render(<SourceView uri={URI} />);

  expect(screen.getByRole('heading', { name: 'report.pdf' })).toBeInTheDocument();
  expect(screen.getByText('rag-datasets/uploads/c1/f1/report.pdf.txt')).toBeInTheDocument();
  expect(await screen.findByText(/Hollywood overtook Europe\./)).toHaveClass('source-page-text');
  expect(screen.getByText('5 words')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith(`/api/v1/chunk-source?uri=${encodeURIComponent(URI)}`);
  expect(document.title).toBe('report.pdf · GraphRAG Search');
});

test('reports a source that cannot be found', async () => {
  global.fetch = jest.fn(() => Promise.resolve({
    ok: true, json: () => Promise.resolve({ found: false, error: 'Failed to fetch source' }),
  }));
  render(<SourceView uri={URI} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Failed to fetch source');
});
