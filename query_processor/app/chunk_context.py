"""
Short context headers embedded in front of each chunk ("contextual chunk headers"), built
without a model call:

  Document: sample | Section: The 1920s | Entities: Hollywood, Europe

At indexing the header has the document name and the nearest section heading. Once the
knowledge graph has extracted a file, its chunks are re-embedded with the names of the
entities they mention added (CorpusIndexer.refresh_vectors). A chunk that only says
"the studios" or "it" then still embeds close to queries that name the entity.

Only the vector uses the header: the stored chunk text (Milvus content, Neo4j content) stays
as written, so citations and quote checks see the original text. all-MiniLM-L6-v2 reads
at most 256 tokens, so the header is kept short (it displaces the end of long chunks).
"""
import re

MAX_SECTION_CHARS = 60
MAX_ENTITIES = 5
MAX_ENTITY_CHARS = 120

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_LINE_END_PUNCT = ".,;:!?\"'”)]"


def _heading_of(paragraph):
    """The heading text if this one-line paragraph looks like a heading, else None: short,
    starting with a capital or digit, not ending like a sentence, not a table row."""
    line = " ".join(paragraph.split())
    if (line and len(line.split()) <= 10 and len(line) <= 80 and " | " not in line
            and line[-1] not in _LINE_END_PUNCT and (line[0].isupper() or line[0].isdigit())):
        return line
    return None


def headings(text):
    """[(offset, heading)]: Markdown '#' lines anywhere, and one-line paragraphs (between
    blank lines) that look like headings, as DOCX and wiki section titles are. A wrapped
    line inside a PDF paragraph is never one, because it isn't a paragraph by itself."""
    out = []
    for m in re.finditer(r"[^\n]+", text):
        md = _MARKDOWN_HEADING.match(m.group(0).strip())
        if md:
            out.append((m.start(), md.group(1)[:MAX_SECTION_CHARS]))
    for m in re.finditer(r"(?:^|\n\s*\n)([^\n]+)(?=\n\s*\n|\s*$)", text):
        heading = _heading_of(m.group(1))
        if heading and not _MARKDOWN_HEADING.match(heading):
            out.append((m.start(1), heading))
    return sorted(out)


def chunk_sections(text, chunks):
    """The section heading for each chunk: the last heading at or before the chunk's start,
    or else the first one inside it ('' when the text has no headings)."""
    found = headings(text)
    sections, cursor = [], 0
    for chunk in chunks:
        start = text.find(chunk[:200], cursor)
        if start < 0:
            start = cursor
        else:
            cursor = start + 1
        before = [h for pos, h in found if pos <= start]
        inside = [h for pos, h in found if start < pos < start + len(chunk)]
        section = before[-1] if before else (inside[0] if inside else "")
        sections.append(section[:MAX_SECTION_CHARS])
    return sections


def make_header(title, section="", entities=()):
    parts = [f"Document: {title}"] if title else []
    if section and section != title:
        parts.append(f"Section: {section}")
    names, length = [], 0
    for name in entities:
        if len(names) >= MAX_ENTITIES or length + len(name) > MAX_ENTITY_CHARS:
            break
        names.append(name)
        length += len(name) + 2
    if names:
        parts.append("Entities: " + ", ".join(names))
    return " | ".join(parts)


def embed_text(content, header=None):
    """The text a chunk's vector is computed from."""
    return f"{header}\n{content}" if header else content
