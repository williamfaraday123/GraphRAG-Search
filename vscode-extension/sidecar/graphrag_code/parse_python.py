"""
The parse result every parser produces (Symbol, Ref, FileResult), and the Python parser that
needs nothing installed: the standard library's ast. It's the fallback for Python when
tree-sitter isn't available (parse_treesitter.py gives the same result; a test checks).

References are kept as written ("helper", "self.save", "models.User", "super.close") and are
resolved to symbols later, across the whole repository (graph.py), so re-parsing one changed
file never needs the others.
"""
import ast
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

MAX_TEXT = 6000  # characters of a symbol's source kept for embeddings and context


@dataclass
class Symbol:
    id: str          # "<path>::<qualname>"; a module is "<path>::"
    path: str        # repository-relative, forward slashes
    kind: str        # module | class | function | method | component | interface | type
    name: str
    qualname: str    # "Class.method", "outer.inner"; "" for a module
    start: int       # first line (1-based), decorators included
    end: int         # last line
    signature: str
    doc: str
    text: str
    parent: str = ""  # id of the enclosing symbol (contains edge)
    sel_line: int = 0  # symbols from a language server: where its name is (1-based line, 0-based column)
    sel_col: int = 0


@dataclass
class Ref:
    src: str         # id of the symbol the reference is in
    kind: str        # call | import | extends | implements | uses_type | render
    target: str      # dotted name as written ("this" is written "self"); "?.name" when the
                     # receiver is unknown; JS/TS imports are "<module path>#<export name>"
    count: int = 1
    line: int = 0


@dataclass
class FileResult:
    path: str
    module: str                      # Python: dotted module name ("pkg.mod"); JS/TS: path without extension
    symbols: List[Symbol] = field(default_factory=list)
    refs: List[Ref] = field(default_factory=list)
    imports: Dict[str, str] = field(default_factory=dict)  # local name -> dotted target / "<path>#<name>"
    error: str = ""                  # syntax error: the file keeps only its module symbol
    lang: str = "python"             # python | typescript | tsx | javascript
    parser: str = "ast"              # which parser made it (a different one re-parses the file)
    exports: Dict[str, str] = field(default_factory=dict)  # JS/TS: "default" -> local name

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> "FileResult":
        return FileResult(
            path=data["path"], module=data["module"],
            symbols=[Symbol(**s) for s in data.get("symbols", [])],
            refs=[Ref(**r) for r in data.get("refs", [])],
            imports=dict(data.get("imports", {})),
            error=data.get("error", ""),
            lang=data.get("lang", "python"),
            parser=data.get("parser", "ast"),
            exports=dict(data.get("exports", {})),
        )


def module_name(path: str) -> str:
    parts = path[:-3].split("/") if path.endswith(".py") else path.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p)


def _dotted(node: ast.AST) -> Optional[str]:
    """'a.b.c' for Name/Attribute chains; 'super.m' for super().m; '?.m' for other receivers."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "super":
            return f"super.{node.attr}"
        inner = _dotted(base)
        return f"{inner}.{node.attr}" if inner and not inner.startswith("?") else f"?.{node.attr}"
    if isinstance(node, ast.Subscript):  # List[User] -> List (annotation names come from walking)
        return _dotted(node.value)
    return None


def _annotation_names(node: Optional[ast.AST]) -> List[str]:
    """Names an annotation refers to, including inside strings ("User", Optional["User"])."""
    if node is None:
        return []
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                names += _annotation_names(ast.parse(sub.value, mode="eval").body)
            except SyntaxError:
                pass
        elif isinstance(sub, (ast.Name, ast.Attribute)):
            dotted = _dotted(sub)
            if dotted and not dotted.startswith("?"):
                names.append(dotted)
    # "typing.Optional" and its parts both appear when walking: keep the longest chains
    return [n for n in dict.fromkeys(names)
            if not any(o != n and o.startswith(n + ".") for o in names)]


def _signature(node: ast.AST) -> str:
    try:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases + node.keywords)
            return f"class {node.name}({bases})" if bases else f"class {node.name}"
    except Exception:
        pass
    return getattr(node, "name", "")


class _Visitor(ast.NodeVisitor):
    def __init__(self, result: FileResult, lines: List[str]):
        self.r, self.lines = result, lines
        self.stack: List[Symbol] = []     # enclosing symbols, innermost last
        self.counts: Dict[tuple, Ref] = {}

    # --- symbols -------------------------------------------------------------------------
    def _text(self, start: int, end: int) -> str:
        return "".join(self.lines[start - 1:end])[:MAX_TEXT]

    def _add_symbol(self, node, kind: str) -> Symbol:
        parent = self.stack[-1]
        qual = f"{parent.qualname}.{node.name}" if parent.qualname else node.name
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        sym = Symbol(
            id=f"{self.r.path}::{qual}", path=self.r.path, kind=kind, name=node.name, qualname=qual,
            start=start, end=node.end_lineno or node.lineno, signature=_signature(node),
            doc=(ast.get_docstring(node) or "")[:500], text=self._text(start, node.end_lineno or node.lineno),
            parent=parent.id,
        )
        self.r.symbols.append(sym)
        return sym

    def _ref(self, kind: str, target: Optional[str], line: int):
        if not target:
            return
        src = self.stack[-1].id
        key = (src, kind, target)
        if key in self.counts:
            self.counts[key].count += 1
        else:
            ref = Ref(src=src, kind=kind, target=target, line=line)
            self.counts[key] = ref
            self.r.refs.append(ref)

    def visit_ClassDef(self, node: ast.ClassDef):
        for d in node.decorator_list:
            self.visit(d)
        sym = self._add_symbol(node, "class")
        self.stack.append(sym)
        for base in node.bases:
            self._ref("extends", _dotted(base), node.lineno)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def _visit_function(self, node):
        for d in node.decorator_list:
            self.visit(d)
        kind = "method" if self.stack[-1].kind == "class" else "function"
        sym = self._add_symbol(node, kind)
        self.stack.append(sym)
        args = node.args
        for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
            if a is not None:
                for name in _annotation_names(a.annotation):
                    self._ref("uses_type", name, node.lineno)
        for name in _annotation_names(node.returns):
            self._ref("uses_type", name, node.lineno)
        for default in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self.visit(default)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # --- references ----------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        self._ref("call", _dotted(node.func), node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        for name in _annotation_names(node.annotation):
            self._ref("uses_type", name, node.lineno)
        if node.value is not None:
            self.visit(node.value)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            # "import a.b" binds "a" but means the module a.b: keep the full name for "a.b.x"
            self.r.imports.setdefault(local, alias.name if alias.asname else alias.name.split(".")[0])
            if not alias.asname and "." in alias.name:
                self.r.imports.setdefault(alias.name, alias.name)
            self._ref("import", alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        base = absolute_module(self.r, node.module or "", node.level)
        for alias in node.names:
            if alias.name == "*":
                self._ref("import", base, node.lineno)
                continue
            target = f"{base}.{alias.name}" if base else alias.name
            self.r.imports.setdefault(alias.asname or alias.name, target)
            self._ref("import", target, node.lineno)


def absolute_module(result: FileResult, module: str, level: int) -> str:
    """'from ..x import y' in pkg/sub/mod.py -> 'pkg.x' (level = number of dots)."""
    if not level:
        return module
    package = result.module.split(".")
    if not result.path.endswith("__init__.py"):
        package = package[:-1]
    package = package[:len(package) - (level - 1)] if level > 1 else package
    return ".".join([p for p in package if p] + ([module] if module else []))


def annotation_names(node) -> List[str]:
    """Public name for _annotation_names (the tree-sitter parser reads string annotations with it)."""
    return _annotation_names(node)


def parse_file(path: str, source: str) -> FileResult:
    """Symbols and references of one file. `path` is repository-relative with '/'."""
    result = FileResult(path=path, module=module_name(path))
    lines = source.splitlines(keepends=True)
    module = Symbol(
        id=f"{path}::", path=path, kind="module", name=path.rsplit("/", 1)[-1], qualname="",
        start=1, end=max(len(lines), 1), signature=result.module, doc="", text="".join(lines[:40])[:MAX_TEXT],
    )
    result.symbols.append(module)
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
        return result
    module.doc = (ast.get_docstring(tree) or "")[:500]
    visitor = _Visitor(result, lines)
    visitor.stack.append(module)
    for stmt in tree.body:
        visitor.visit(stmt)
    return result
