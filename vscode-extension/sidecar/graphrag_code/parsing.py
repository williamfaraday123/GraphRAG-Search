"""
Which parser reads a file: tree-sitter (Python, TypeScript, TSX, JavaScript) when it's
installed and loads, else the standard library's ast (Python only).
"""
import ast
import posixpath
from typing import Optional

from . import parse_treesitter as ts
from .parse_python import FileResult, parse_file as parse_with_ast

_mode = "auto"  # "auto": tree-sitter if available; "ast": Python only, with ast


def set_mode(mode: str) -> None:
    global _mode
    _mode = mode if mode in ("auto", "ast") else "auto"


def using_treesitter() -> bool:
    return _mode == "auto" and ts.available()


def parser_name() -> str:
    return ts.PARSER_NAME if using_treesitter() else "ast"


def supported(path: str) -> bool:
    if using_treesitter():
        return ts.language_of(path) is not None
    return path.endswith(".py")


def parse_source(path: str, source: str) -> Optional[FileResult]:
    if using_treesitter():
        return ts.parse(path, source)
    return parse_with_ast(path, source) if path.endswith(".py") else None


def syntax_error(path: str, source: str) -> Optional[str]:
    """'path:line: message' when the file doesn't parse, else None."""
    if path.endswith(".py"):
        try:
            ast.parse(source, filename=path)
        except SyntaxError as e:
            return f"{path}:{e.lineno}: {e.msg}"
        return None
    if ts.available() and ts.language_of(path):
        line = ts.first_error_line(source, ts.language_of(path))
        return f"{path}:{line}: syntax error" if line else None
    return None


def parsed_extensions() -> list:
    """File extensions a parser reads; language servers cover the rest."""
    return sorted(ts.EXTENSIONS) if using_treesitter() else [".py"]


def code_language(path: str) -> str:
    """For Markdown code fences."""
    lang = ts.EXTENSIONS.get(posixpath.splitext(path)[1].lower(), "")
    return {"tsx": "tsx", "typescript": "typescript", "javascript": "javascript"}.get(lang, "python")


def status() -> dict:
    return {
        "parser": parser_name(),
        "supportedLanguages": ["python", "typescript", "tsx", "javascript"] if using_treesitter() else ["python"],
        "parserProblem": "" if ts.available() else (ts.IMPORT_ERROR or "tree-sitter is not installed"),
    }
