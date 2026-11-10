import { render } from '@testing-library/react';
import Markdown, { parseOptionChecks, splitAnswer } from './Markdown';

const html = (text) => render(<Markdown text={text} />).container.firstChild;

test('renders bold, italic, code and file citations instead of raw markers', () => {
  const el = html('Hollywood had **more capital** and *better* `narratives` [sample.docx, sample2.pdf].');
  expect(el.querySelector('strong')).toHaveTextContent('more capital');
  expect(el.querySelector('em')).toHaveTextContent('better');
  expect(el.querySelector('code')).toHaveTextContent('narratives');
  expect([...el.querySelectorAll('.md-cite')].map((c) => c.textContent)).toEqual(['sample.docx', 'sample2.pdf']);
  expect(el.textContent).not.toMatch(/\*|`/);
});

test('renders headings, paragraphs, nested lists, quotes and tables', () => {
  const el = html([
    '## Summary',
    'First line',
    'continues here.',
    '',
    '1. **Capital** to buy talent',
    '   - hired European directors',
    '2. Better narratives',
    '',
    '> "Hollywood took the lead." [sample.docx]',
    '',
    '| Option | Result |',
    '| --- | --- |',
    '| 1 | SUPPORTED |',
    '| 3 | NOT SUPPORTED |',
  ].join('\n'));
  expect(el.querySelector('h3')).toHaveTextContent('Summary');
  expect(el.querySelector('p')).toHaveTextContent('First line continues here.');
  const ol = el.querySelector('ol');
  expect(ol.children).toHaveLength(2);
  expect(ol.querySelector('li ul li')).toHaveTextContent('hired European directors');
  expect(el.querySelector('blockquote')).toHaveTextContent('"Hollywood took the lead." sample.docx');
  expect([...el.querySelectorAll('tbody tr')].map((r) => r.textContent)).toEqual(['1SUPPORTED', '3NOT SUPPORTED']);
});

test('model text cannot inject HTML', () => {
  const el = html('<img src=x onerror=alert(1)> **bold**');
  expect(el.querySelector('img')).toBeNull();
  expect(el).toHaveTextContent('<img src=x onerror=alert(1)> bold');
});

test('splits the checked evidence from the answer; other replies come back whole', () => {
  const reply = '### Checked evidence\n- **1**: SUPPORTED. "x" [a.pdf]\n\n### Answer\n**1 and 4** [a.pdf]';
  expect(splitAnswer(reply)).toEqual({ answer: '**1 and 4** [a.pdf]', checks: '- **1**: SUPPORTED. "x" [a.pdf]' });
  expect(splitAnswer('Final answer: 1')).toEqual({ answer: 'Final answer: 1', checks: '' });
  // Still streaming: no Answer heading yet
  expect(splitAnswer('### Checked evidence\n- **1**: SUPP').checks).toBe('');
});

test('bold containing italics, bold italics and unclosed bold leave no asterisks', () => {
  const el = html('**bold with *italic* inside**, ***both*** and **unclosed bold');
  expect(el.querySelector('strong')).toHaveTextContent('bold with italic inside');
  expect(el.querySelector('strong em')).toHaveTextContent('italic');
  expect([...el.querySelectorAll('strong')].map((s) => s.textContent)).toContain('both');
  expect(el.textContent).toBe('bold with italic inside, both and unclosed bold');
});

const replies = require('./__fixtures__/modelReplies.json');

test('parses a real multiple-choice reply into every option with its verdict, in option order', () => {
  const { answer, checks } = splitAnswer(replies.bothFiles);
  expect(answer).toMatch(/^\*\*Answer: 4 \(well-written narratives\)\*\*/);
  const options = parseOptionChecks(checks);
  expect(options.map((o) => `${o.marker} ${o.supported ? 'yes' : 'no'}`)).toEqual(['1 no', '2 no', '3 no', '4 yes', '5 no']);
  const four = options[3];
  expect(four.label).toBe('well-written narratives');
  expect(four.evidence).toMatch(/^\[sample\.docx\] "Hollywood films appealed/);
  expect(four.evidence).toContain('[sample2.pdf]');
  expect(four.reason).not.toMatch(/^[:.]/);
  expect(parseOptionChecks(splitAnswer(replies.twoReasons).checks).filter((o) => o.supported).map((o) => o.marker))
    .toEqual(['1', '4']);
});

test('a reply without section headings still splits, and options are sorted by number', () => {
  const { answer, checks } = splitAnswer(replies.noHeadings);
  expect(answer).toMatch(/^\*\*Answer: 1/);
  expect(parseOptionChecks(checks).map((o) => o.marker)).toEqual(['1', '2', '3', '4', '5']);
});

test('open questions are not treated as option breakdowns', () => {
  expect(parseOptionChecks(splitAnswer(replies.open).checks)).toBeNull();
  expect(parseOptionChecks('- **plenty of capital**: SUPPORTED. "x" [a.pdf]\n- **b**: NOT SUPPORTED.')).toBeNull();
});

test('"[file 1: report.pdf]" citations show just the file name', () => {
  expect(html('See [file 1: report.pdf].').querySelector('.md-cite')).toHaveTextContent(/^report\.pdf$/);
});

test('two trailing spaces make a line break; a plain newline joins lines', () => {
  const el = html('**Answer: 4 (narratives)**  \nBoth files say so.\nSame paragraph.');
  expect(el.querySelectorAll('p')).toHaveLength(1);
  expect(el.querySelectorAll('br')).toHaveLength(1);
  expect(el.querySelector('p')).toHaveTextContent('Answer: 4 (narratives)Both files say so. Same paragraph.');
});
