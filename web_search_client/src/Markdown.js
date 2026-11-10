import React from 'react';

// A small Markdown renderer for model answers: headings, paragraphs, bold/italic/code,
// nested lists, blockquotes, pipe tables, code fences and rules. It builds React elements
// (never HTML strings), so model output can't inject markup. File citations such as
// [report.pdf] are shown as tags.

const FILE_EXT = '(?:txt|md|pdf|docx|mp3)';
const INLINE = new RegExp(
  [
    '`[^`]+`', // code
    '\\*\\*\\*(?!\\s)[^*]+?\\*\\*\\*', // ***bold italic***
    '\\*\\*(?!\\s)(?:[^*]|\\*(?!\\*))+?\\*\\*', // **bold**, may contain *italic*
    '__(?!\\s)[^_]+?__', // __bold__
    '(?<![\\w*])\\*(?!\\s)[^*]+?(?<!\\s)\\*(?![\\w*])', // *italic*
    `\\[[^\\]\\n]*?\\.${FILE_EXT}(?:\\s*[,;]\\s*[^\\]\\n]*?\\.${FILE_EXT})*\\]`, // [file.pdf] citations
  ].join('|'),
  'gi'
);

// Bold markers left over once pairs are matched (the model left one unclosed, or its output
// was cut off): shown as nothing rather than as literal asterisks
const plain = (text) => text.replace(/\*\*/g, '');

export function renderInline(text, keyPrefix = 'i') {
  const out = [];
  let last = 0;
  let n = 0;
  for (const match of text.matchAll(INLINE)) {
    const token = match[0];
    if (match.index > last) out.push(plain(text.slice(last, match.index)));
    const key = `${keyPrefix}-${n++}`;
    if (token.startsWith('`')) {
      out.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('***')) {
      out.push(<strong key={key}><em>{renderInline(token.slice(3, -3), key)}</em></strong>);
    } else if (token.startsWith('**') || token.startsWith('__')) {
      out.push(<strong key={key}>{renderInline(token.slice(2, -2), key)}</strong>);
    } else if (token.startsWith('*')) {
      out.push(<em key={key}>{renderInline(token.slice(1, -1), key)}</em>);
    } else {
      // [a.pdf] or [a.pdf, b.docx]: one tag per file
      token.slice(1, -1).split(/\s*[,;]\s*/).forEach((name, j) => {
        // "[file 1: report.pdf]" -> report.pdf
        const file = name.trim().replace(/^file\s*\d+\s*:\s*/i, '');
        out.push(<span key={`${key}-${j}`} className="md-cite">{file}</span>);
      });
    }
    last = match.index + token.length;
  }
  if (last < text.length) out.push(plain(text.slice(last)));
  return out;
}

const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const FENCE = /^\s*```/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;
const isTableRow = (line) => line.includes('|');
const cells = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
const startsBlock = (line, next) => HEADING.test(line) || RULE.test(line) || LIST_ITEM.test(line)
  || FENCE.test(line) || /^\s*>/.test(line) || (isTableRow(line) && next !== undefined && TABLE_DIVIDER.test(next));

function parseList(lines, start, key) {
  // Items at the first item's indent; deeper-indented items nest under the previous item
  const first = lines[start].match(LIST_ITEM);
  const indent = first[1].length;
  const ordered = /\d/.test(first[2]);
  const items = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // A blank line ends the list unless the next line continues it
      const next = lines[i + 1];
      const m = next && next.match(LIST_ITEM);
      if (m && m[1].length >= indent) { i += 1; continue; }
      break;
    }
    const m = line.match(LIST_ITEM);
    if (m && m[1].length === indent) {
      items.push({ text: m[3], children: [] });
      i += 1;
    } else if (m && m[1].length > indent && items.length) {
      const [nested, end] = parseList(lines, i, `${key}-${items.length}`);
      items[items.length - 1].children.push(nested);
      i = end;
    } else if (!m && /^\s+\S/.test(line) && items.length) {
      items[items.length - 1].text += ` ${line.trim()}`; // continuation line
      i += 1;
    } else {
      break;
    }
  }
  const Tag = ordered ? 'ol' : 'ul';
  return [
    <Tag key={key}>
      {items.map((item, j) => (
        <li key={j}>{renderInline(item.text, `${key}-${j}`)}{item.children}</li>
      ))}
    </Tag>,
    i,
  ];
}

export function parseBlocks(text, keyPrefix = 'b') {
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const key = `${keyPrefix}-${blocks.length}`;
    if (!line.trim()) { i += 1; continue; }

    const heading = line.match(HEADING);
    if (heading) {
      // Inside an answer, Markdown h1-h2 render as h3; deeper levels step down from there
      const Tag = `h${Math.min(6, Math.max(3, heading[1].length + 1))}`;
      blocks.push(<Tag key={key}>{renderInline(heading[2], key)}</Tag>);
      i += 1;
    } else if (RULE.test(line)) {
      blocks.push(<hr key={key} />);
      i += 1;
    } else if (FENCE.test(line)) {
      const body = [];
      i += 1;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      i += 1;
      blocks.push(<pre key={key}><code>{body.join('\n')}</code></pre>);
    } else if (/^\s*>/.test(line)) {
      const body = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ''));
      blocks.push(<blockquote key={key}>{parseBlocks(body.join('\n'), key)}</blockquote>);
    } else if (isTableRow(line) && TABLE_DIVIDER.test(lines[i + 1] || '')) {
      const head = cells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && isTableRow(lines[i])) rows.push(cells(lines[i++]));
      blocks.push(
        <div key={key} className="md-table">
          <table>
            <thead><tr>{head.map((c, j) => <th key={j}>{renderInline(c, `${key}-h${j}`)}</th>)}</tr></thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r}>{head.map((_, j) => <td key={j}>{renderInline(row[j] || '', `${key}-${r}-${j}`)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    } else if (LIST_ITEM.test(line)) {
      const [list, end] = parseList(lines, i, key);
      blocks.push(list);
      i = end;
    } else {
      // A line ending in two spaces or a backslash is a hard line break ("**Answer: 4**  \nWhy ...")
      const body = [line];
      i += 1;
      while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) body.push(lines[i++]);
      const parts = [];
      body.forEach((raw, j) => {
        if (j > 0) parts.push(/( {2,}|\\)$/.test(body[j - 1]) ? <br key={`${key}-br${j}`} /> : ' ');
        parts.push(...renderInline(raw.trim().replace(/\\$/, ''), `${key}-${j}`));
      });
      blocks.push(<p key={key}>{parts}</p>);
    }
  }
  return blocks;
}

export default function Markdown({ text, className = '' }) {
  return <div className={`markdown ${className}`.trim()}>{parseBlocks(text || '')}</div>;
}

const CHECKS_HEADING = /^#{1,6}\s*(?:checked evidence|evidence checked|evidence|checks?)\s*:?\s*$/im;
const ANSWER_HEADING = /^#{1,6}\s*(?:final\s+)?answer\s*:?\s*$/im;

/**
 * Splits a model reply written as "### Checked evidence ... ### Answer ..." so the UI can
 * show the answer first and the checks collapsed. Replies without an Answer heading
 * (other formats, or still streaming) come back whole as the answer.
 */
export function splitAnswer(text) {
  const answerAt = text.search(ANSWER_HEADING);
  if (answerAt < 0) {
    // Headings left out: option checks as bullets, then a bold "**Answer: ...**" line
    const boldAnswer = text.search(/^\*\*Answer\b/im);
    if (boldAnswer > 0 && parseOptionChecks(text.slice(0, boldAnswer))) {
      return { answer: text.slice(boldAnswer).trim(), checks: text.slice(0, boldAnswer).replace(CHECKS_HEADING, '').trim() };
    }
    return { answer: text, checks: '' };
  }
  const afterHeading = text.slice(answerAt).replace(ANSWER_HEADING, '');
  const before = text.slice(0, answerAt);
  const checks = before.replace(CHECKS_HEADING, '').trim();
  return { answer: afterHeading.trim(), checks };
}

// "- **2. lower rents**: [survey.pdf] "quote" NOT SUPPORTED: only one file says it."
const OPTION_BULLET = /^\s*[-*+]\s+\*\*\s*([1-9][0-9]?|[A-Ha-h])[.)]\s*(.+?)\*\*\s*[:\-–—]?\s*(.*)$/;
const VERDICT = /\b(NOT\s+SUPPORTED|SUPPORTED)\b\s*[:.\-–—]?\s*/i;

/**
 * The model's per-option checks for a multiple-choice question, as
 * [{ marker, label, supported, evidence, reason }] sorted by option, or null when the text
 * isn't an option breakdown (open questions, or a reply in another format).
 */
export function parseOptionChecks(checks) {
  const lines = (checks || '').split('\n').filter((l) => l.trim());
  const items = [];
  for (const line of lines) {
    const m = line.match(OPTION_BULLET);
    if (!m) {
      if (/^\s*[-*+]\s/.test(line)) return null; // a bullet that isn't an option check
      if (items.length) items[items.length - 1].rest += ` ${line.trim()}`; // continuation
      continue;
    }
    items.push({ marker: m[1].toUpperCase(), label: m[2].trim(), rest: m[3] });
  }
  if (items.length < 2) return null;
  const parsed = items.map(({ marker, label, rest }) => {
    const verdict = rest.match(VERDICT);
    if (!verdict) return null;
    const reason = rest.slice(verdict.index + verdict[0].length).trim();
    return {
      marker,
      label,
      supported: !/^NOT/i.test(verdict[1]),
      evidence: rest.slice(0, verdict.index).trim().replace(/[.;]\s*$/, ''),
      reason: reason.charAt(0).toUpperCase() + reason.slice(1),
    };
  });
  if (parsed.some((p) => !p)) return null;
  const order = (m) => (/\d/.test(m) ? Number(m) : m.charCodeAt(0));
  return parsed.sort((a, b) => order(a.marker) - order(b.marker));
}
