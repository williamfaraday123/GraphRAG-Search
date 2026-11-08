"""
Turns uploaded files into plain text for chunking, preview and "view full source".

Uploads are converted to text once, at upload, and only that text is stored, as
uploads/<corpus>/<file_id>/<original name>.txt (stored_text_name). The original PDF,
Word or audio file is discarded, which saves space and means preview/view-source just
read UTF-8 text. display_name() turns the stored name back into the original name.

read_text_object() still understands the older layout, where the original file was
stored as-is (and audio had a separate transcript object), so earlier uploads keep working.
"""
import io
import logging
import re
from pathlib import PurePath

logger = logging.getLogger(__name__)

STORED_TEXT_SUFFIX = ".txt"
STORED_CONTENT_TYPE = "text/plain; charset=utf-8"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".mp3"}
AUDIO_EXTENSIONS = {".mp3"}


def extension_of(filename: str) -> str:
    return PurePath(filename or "").suffix.lower()


def is_audio(filename: str) -> bool:
    return extension_of(filename) in AUDIO_EXTENSIONS


def stored_text_name(filename: str) -> str:
    """report.pdf -> report.pdf.txt, notes.txt -> notes.txt.txt (always appended, so
    display_name can reverse it unambiguously)."""
    return filename + STORED_TEXT_SUFFIX


def display_name(stored_name: str) -> str:
    """Reverses stored_text_name. Names from the older layout (original stored as-is)
    are returned unchanged."""
    if stored_name.lower().endswith(STORED_TEXT_SUFFIX):
        original = stored_name[: -len(STORED_TEXT_SUFFIX)]
        if extension_of(original) in SUPPORTED_EXTENSIONS:
            return original
    return stored_name


def transcript_key_for(object_key: str) -> str:
    """Older layout only: uploads/<corpus>/<file_id>/<name>.mp3 -> transcripts/<corpus>/<file_id>.txt"""
    parts = object_key.split("/")
    if len(parts) != 4 or parts[0] != "uploads":
        raise ValueError(f"Not an upload object key: {object_key}")
    _, corpus_id, file_id, _ = parts
    return f"transcripts/{corpus_id}/{file_id}.txt"


def read_text_object(s3, bucket: str, key: str) -> str:
    """Returns the text of a stored object: UTF-8 text for current uploads and wiki
    sources, or extracted text for uploads stored in their original format."""
    if is_audio(key):  # older layout: transcript stored separately
        return s3.get_object(Bucket=bucket, Key=transcript_key_for(key))["Body"].read().decode("utf-8")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if extension_of(key) in SUPPORTED_EXTENSIONS:  # .txt/.md decode; older .pdf/.docx extract
        return extract_text(key, body)
    return body.decode("utf-8")


def extract_text(filename: str, content: bytes) -> str:
    """Returns the file's text. Raises ValueError with a user-facing message when the
    file can't be read (wrong format, encrypted PDF, corrupt file...)."""
    ext = extension_of(filename)
    if ext in (".txt", ".md"):
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("File must contain valid UTF-8 text") from exc
    if ext == ".pdf":
        return _pdf_text(content)
    if ext == ".docx":
        return _docx_text(content)
    if ext in AUDIO_EXTENSIONS:
        from transcriber import transcribe  # heavy import; only needed for audio
        return transcribe(content)
    raise ValueError("Only .txt, .md, .pdf, .docx and .mp3 files are supported")


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported")
        # Blank line between pages so the chunker can split on page boundaries
        pages = [_pdf_page_text(page) for page in reader.pages]
    except ValueError:
        raise
    except Exception as exc:  # pypdf raises many types for malformed files
        logger.warning("Could not read PDF: %s", exc)
        raise ValueError("Could not read this PDF; it may be damaged") from exc

    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise ValueError(
            "No text found in this PDF. Scanned or image-only PDFs need OCR, which is not supported"
        )
    return text


def _is_fragmented(text: str) -> bool:
    """True when most lines hold a single word: some PDF writers (e.g. Google Docs'
    Skia renderer) place every word and space separately, and pypdf's default
    extraction then puts each one on its own line."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        return False
    single_word = sum(1 for line in lines if len(line.split()) <= 1)
    return single_word / len(lines) > 0.6


def _clean_layout_text(text: str) -> str:
    """Layout-mode text pads with runs of spaces to keep columns aligned. Collapse them,
    and keep blank lines only as single paragraph breaks."""
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _pdf_page_text(page) -> str:
    text = (page.extract_text() or "").strip()
    if not _is_fragmented(text):
        return text
    # Layout mode rebuilds lines from glyph positions, so word-by-word PDFs read normally.
    # Used only as a fallback: on multi-column pages it would merge the columns.
    try:
        layout = _clean_layout_text(page.extract_text(extraction_mode="layout") or "")
    except Exception as exc:
        logger.warning("Layout extraction failed, using default PDF text: %s", exc)
        return text
    return layout or text


def _docx_text(content: bytes) -> str:
    import docx
    from docx.document import Document as DocumentType
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document: DocumentType = docx.Document(io.BytesIO(content))
    except Exception as exc:
        logger.warning("Could not read DOCX: %s", exc)
        raise ValueError("Could not read this Word file; only .docx (not .doc) is supported") from exc

    blocks = []
    # Walk the body in document order so tables stay next to the text that introduces them
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            if item.text.strip():
                blocks.append(item.text.strip())
        elif isinstance(item, Table):
            for row in item.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip()
                    if value and (not cells or cells[-1] != value):  # merged cells repeat
                        cells.append(value)
                if cells:
                    blocks.append(" | ".join(cells))
    return "\n\n".join(blocks)
