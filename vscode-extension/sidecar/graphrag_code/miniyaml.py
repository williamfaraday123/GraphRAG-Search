"""
Enough YAML for docker-compose files, standard library only (PyYAML is used when installed):
block mappings and sequences by indentation, `- key: value` items, inline [a, b] and {a: b},
quoted and plain scalars, comments. Block scalars (| and >) are read as their joined lines.
Anchors, tags and multi-document files aren't supported (and aren't needed for what is read).
"""
import re
from typing import Any, List, Tuple

_KEY = re.compile(r"""^(?:"([^"]*)"|'([^']*)'|([^\s#'"][^:#]*?))\s*:(?:\s+|$)(.*)$""")


def load(text: str) -> Any:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        pass
    except Exception:
        return None
    lines = []
    for raw in text.splitlines():
        if raw.strip() in ("---", "..."):
            continue
        stripped = _strip_comment(raw).rstrip()
        if stripped.strip():
            lines.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))
    try:
        value, _ = _block(lines, 0, lines[0][0] if lines else 0)
        return value
    except (IndexError, ValueError):
        return None


def _strip_comment(line: str) -> str:
    quote = None
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _block(lines: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    if i >= len(lines):
        return None, i
    if lines[i][1].startswith("- ") or lines[i][1] == "-":
        return _sequence(lines, i, indent)
    return _mapping(lines, i, indent)


def _sequence(lines, i, indent):
    out = []
    while i < len(lines) and lines[i][0] == indent and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
        rest = lines[i][1][1:].strip()
        if not rest:  # "-" then a nested block
            value, i = _block(lines, i + 1, lines[i + 1][0]) if i + 1 < len(lines) and lines[i + 1][0] > indent else (None, i + 1)
            out.append(value)
            continue
        m = _KEY.match(rest)
        if m and not rest.startswith(("[", "{")):
            # "- key: value" starts a mapping whose other keys are indented under it
            child = indent + 2
            sub = [(child, rest)]
            j = i + 1
            while j < len(lines) and lines[j][0] > indent:
                sub.append(lines[j])
                j += 1
            value, _ = _mapping(sub, 0, child)
            out.append(value)
            i = j
            continue
        out.append(_scalar(rest))
        i += 1
    return out, i


def _mapping(lines, i, indent):
    out = {}
    while i < len(lines) and lines[i][0] == indent:
        text = lines[i][1]
        m = _KEY.match(text)
        if not m:
            raise ValueError(text)
        key = next(g for g in m.groups()[:3] if g is not None).strip()
        rest = m.group(4).strip()
        i += 1
        if rest in ("|", ">", "|-", ">-", "|+", ">+"):
            parts = []
            while i < len(lines) and lines[i][0] > indent:
                parts.append(lines[i][1])
                i += 1
            out[key] = ("\n" if rest.startswith("|") else " ").join(parts)
        elif rest:
            out[key] = _scalar(rest)
        elif i < len(lines) and (lines[i][0] > indent or (lines[i][0] == indent and lines[i][1].startswith("- "))):
            out[key], i = _block(lines, i, lines[i][0])
        else:
            out[key] = None
    return out, i


def _split_inline(body: str) -> List[str]:
    parts, depth, quote, cur = [], 0, None, ""
    for c in body:
        if quote:
            cur += c
            if c == quote:
                quote = None
            continue
        if c in "'\"":
            quote = c
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _scalar(text: str) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(p) for p in _split_inline(text[1:-1])]
    if text.startswith("{") and text.endswith("}"):
        out = {}
        for p in _split_inline(text[1:-1]):
            k, _, v = p.partition(":")
            out[str(_scalar(k))] = _scalar(v) if v.strip() else None
        return out
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text
