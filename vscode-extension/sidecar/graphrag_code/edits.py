"""
Edits as the model writes them, and turning them into new file contents.

The model rewrites whole symbols rather than writing a diff, which small models get right far
more often:

    === REPLACE path/to/file.py :: Class.method
    <the complete new source of that symbol, decorators included>
    === END
    === INSERT_AFTER path/to/file.py :: helper
    <new code placed right after that symbol>
    === END
    === APPEND path/to/file.py
    <new code added at the end of the file>
    === END
    === CREATE path/to/new_file.py
    <the whole new file>
    === END

Symbols are located by re-parsing the file as it is now, code is re-indented to where it goes,
and every changed file must still parse before it is proposed.
"""
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import parsing

_EXPORT = re.compile(r"^(\s*)(export\s+(?:default\s+)?)")


def _symbols(path: str, text: str, symbols_of=None):
    """The file's symbols: from a parser, else (files only a language server reads) from
    `symbols_of(path, text)`, which knows them for the file as it is on disk; None if unknown."""
    result = parsing.parse_source(path, text)
    if result is not None:
        return result.symbols
    return symbols_of(path, text) if symbols_of else None


def _keep_export(original_first_line: str, new_lines: List[str]) -> List[str]:
    """JS/TS: a symbol that was exported stays exported when the model leaves `export` out."""
    m = _EXPORT.match(original_first_line)
    if not m:
        return new_lines
    for i, line in enumerate(new_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("@"):
            continue
        if not stripped.startswith("export"):
            indent = line[:len(line) - len(line.lstrip())]
            new_lines[i] = f"{indent}{m.group(2)}{line.lstrip()}"
        break
    return new_lines

_HEADER = re.compile(r"^===\s*(REPLACE|INSERT_AFTER|APPEND|CREATE)\s+(\S+?)(?:\s*::\s*(\S+))?\s*$")
_END = re.compile(r"^===\s*END\s*$")
_LINES = re.compile(r"^(.+?\.[A-Za-z]{1,4}):(\d+)(?:-\d+)?$")
_FENCE = re.compile(r"^\s*```")


@dataclass
class EditBlock:
    op: str
    path: str
    symbol: str
    code: str
    line: int = 0    # "path:18-19" written instead of "path :: name": the symbol at that line


@dataclass
class Proposal:
    files: Dict[str, dict] = field(default_factory=dict)   # path -> {original, proposed, created}
    changed_symbols: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)       # blocks that could not be applied
    syntax_errors: List[str] = field(default_factory=list)


def parse_blocks(text: str) -> List[EditBlock]:
    blocks, current, lines = [], None, []
    for line in text.splitlines():
        header = _HEADER.match(line.strip())
        if header:
            if current:  # a block without END: close it at the next header
                blocks.append(_finish(current, lines))
            current, lines = header.groups(), []
            continue
        if current and _END.match(line.strip()):
            blocks.append(_finish(current, lines))
            current, lines = None, []
            continue
        if current:
            lines.append(line)
    if current:
        blocks.append(_finish(current, lines))
    return blocks


def _finish(header, lines) -> EditBlock:
    op, path, symbol = header
    # Small models wrap code in ``` fences even when told not to
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if lines and _FENCE.match(lines[0]):
        lines = lines[1:]
    if lines and _FENCE.match(lines[-1]):
        lines = lines[:-1]
    path = path.strip("`").lstrip("./")
    where = _LINES.match(path)
    line = 0
    if where:  # "pkg/models.py:18-19"
        path, line = where.group(1), int(where.group(2))
    return EditBlock(op=op, path=path, symbol=(symbol or "").strip("`"), code="\n".join(lines), line=line)


def _indent_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _place(code: str, indent: str) -> List[str]:
    body = textwrap.dedent(code).strip("\n")
    return [(indent + l if l.strip() else "") + "\n" for l in body.split("\n")]


def apply_blocks(blocks: List[EditBlock], read: Callable[[str], Optional[str]], symbols_of=None) -> Proposal:
    """`read(path)` gives a file's current text (None if it doesn't exist); `symbols_of(path,
    text)` gives the symbols of files no parser reads (see _symbols)."""
    proposal = Proposal()
    by_file: Dict[str, List[EditBlock]] = {}
    for b in blocks:
        by_file.setdefault(b.path, []).append(b)

    for path, file_blocks in by_file.items():
        original = read(path)
        if any(b.op == "CREATE" for b in file_blocks):
            if original is not None:
                proposal.skipped.append(f"CREATE {path}: the file already exists")
                continue
            code = next(b.code for b in file_blocks if b.op == "CREATE")
            proposed = textwrap.dedent(code).strip("\n") + "\n"
            proposal.files[path] = {"original": "", "proposed": proposed, "created": True}
            proposal.changed_symbols += [s.id for s in (_symbols(path, proposed) or []) if s.kind != "module"]
            continue
        if original is None:
            proposal.skipped += [f"{b.op} {path}: no such file" for b in file_blocks]
            continue
        lines = original.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        parsed = [s for s in (_symbols(path, original, symbols_of) or []) if s.qualname]
        symbols = {s.qualname: s for s in parsed}
        # (start index, end index, new lines); applied bottom-up so line numbers stay valid
        splices = []
        replaced = []  # (symbol, its old code, its new code): for files no parser reads
        for b in file_blocks:
            if b.op == "APPEND":
                splices.append((len(lines), len(lines), ["\n"] + _place(b.code, "")))
                continue
            sym = symbols.get(b.symbol) or _by_short_name(symbols, b.symbol) or _at_line(parsed, b.line)
            if sym is None:
                proposal.skipped.append(f"{b.op} {path} :: {b.symbol or b.line}: no such symbol")
                continue
            indent = _indent_of(lines[sym.start - 1]) if sym.start - 1 < len(lines) else ""
            if b.op == "REPLACE":
                # Counted as changed below only if its code really differs: small models
                # often "replace" symbols with the same code
                new = _keep_export(lines[sym.start - 1], _place(b.code, indent))
                splices.append((sym.start - 1, sym.end, new))
                replaced.append((sym, "".join(lines[sym.start - 1:sym.end]), "".join(new)))
            else:  # INSERT_AFTER
                splices.append((sym.end, sym.end, ["\n"] + _place(b.code, indent)))
        if not splices:
            continue
        splices.sort(key=lambda s: (s[0], s[1]), reverse=True)
        for start, end, new in splices:
            lines[start:end] = new
        proposed = "".join(lines)
        if proposed == original:
            continue
        proposal.files[path] = {"original": original, "proposed": proposed, "created": False}
        # Symbols that are new or whose own code changed count as changed (impact analysis);
        # a class whose only change is inside a method doesn't
        after = _symbols(path, proposed)
        if after is None:  # no parser for this file: the replaced symbols whose code differs
            for sym, old, new in replaced:
                if old.strip() != new.strip() and sym.id not in proposal.changed_symbols:
                    proposal.changed_symbols.append(sym.id)
            continue
        before = _own_texts(_symbols(path, original) or [])
        for qual, text in _own_texts(after).items():
            sid = f"{path}::{qual}"
            if before.get(qual) != text and sid not in proposal.changed_symbols:
                proposal.changed_symbols.append(sid)

    for path, f in proposal.files.items():
        problem = parsing.syntax_error(path, f["proposed"])
        if problem:
            proposal.syntax_errors.append(problem)
    return proposal


def _own_texts(symbols) -> Dict[str, str]:
    """{qualname: its code with the code of the symbols nested in it left out}."""
    out = {}
    for s in symbols:
        if not s.qualname:
            continue
        text = s.text
        for child in symbols:
            if child.parent == s.id and child.text:
                text = text.replace(child.text, f"<{child.qualname}>")
        out[s.qualname] = text
    return out


def _at_line(symbols, line: int):
    """The symbol starting at `line`, else the innermost one containing it."""
    if not line:
        return None
    starting = [s for s in symbols if s.start == line]
    if starting:
        return max(starting, key=lambda s: s.end - s.start)
    containing = [s for s in symbols if s.start <= line <= s.end]
    return min(containing, key=lambda s: s.end - s.start) if containing else None


def _by_short_name(symbols: Dict[str, object], name: str):
    """'method' for 'Class.method' when the model left out the class (and it's unambiguous)."""
    if not name:
        return None
    hits = [s for q, s in symbols.items() if q.split(".")[-1] == name.split(".")[-1]]
    return hits[0] if len(hits) == 1 else None
