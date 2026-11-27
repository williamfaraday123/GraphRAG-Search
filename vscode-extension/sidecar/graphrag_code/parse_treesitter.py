"""
Parsing with tree-sitter: Python, TypeScript, TSX and JavaScript/JSX -> the same symbols and
references as parse_python.py (static analysis, no model).

Python gives exactly what the ast parser gives (a test compares them), so the two can stand in
for each other. TypeScript/JavaScript adds:
  symbols     functions (also `const f = () => ...`, and React.memo/forwardRef wrappers),
              classes, methods (and arrow-function class fields), interfaces, type aliases;
              a capitalised function that returns JSX is a `component`
  references  calls and `new`, JSX elements (`render`: component -> child component),
              `extends`, `implements`, type annotations (`uses_type`), imports
  imports     relative ones only ("./ui/Button" -> "src/ui/Button#default"); packages and
              path aliases aren't in the repository. Re-exports (`export * from`,
              `export { default as Button } from`) resolve through barrel files.

A symbol's lines include its `export` keyword, so replacing a symbol keeps it exported.
"""
import ast
import inspect
import posixpath
import re
from typing import Dict, List, Optional

from .parse_python import (MAX_TEXT, FileResult, Ref, Symbol, absolute_module, annotation_names,
                           module_name)

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript
    _GRAMMARS = {
        "python": lambda: tree_sitter_python.language(),
        "typescript": lambda: tree_sitter_typescript.language_typescript(),
        "tsx": lambda: tree_sitter_typescript.language_tsx(),
        "javascript": lambda: tree_sitter_javascript.language(),
    }
    IMPORT_ERROR = ""
except Exception as e:  # not installed, or its compiled module was blocked
    _GRAMMARS = {}
    IMPORT_ERROR = f"{type(e).__name__}: {e}"

PARSER_NAME = "tree-sitter"
EXTENSIONS = {".py": "python", ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
              ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
              ".cjs": "javascript"}
JS_EXTENSIONS = [e for e, lang in EXTENSIONS.items() if lang != "python"]
_parsers: Dict[str, "Parser"] = {}


def available() -> bool:
    return bool(_GRAMMARS)


def language_of(path: str) -> Optional[str]:
    if path.endswith(".d.ts") or path.endswith(".min.js"):
        return None
    return EXTENSIONS.get(posixpath.splitext(path)[1].lower())


def _parser(lang: str) -> "Parser":
    if lang not in _parsers:
        _parsers[lang] = Parser(Language(_GRAMMARS[lang]()))
    return _parsers[lang]


def strip_extension(path: str) -> str:
    root, ext = posixpath.splitext(path)
    return root if ext.lower() in EXTENSIONS else path


def first_error_line(source: str, lang: str) -> Optional[int]:
    """Line (1-based) of the first syntax error, or None when the file parses."""
    tree = _parser(lang).parse(source.encode("utf-8"))
    if not tree.root_node.has_error:
        return None
    stack = [tree.root_node]
    while stack:
        node = stack.pop(0)
        if node.type == "ERROR" or node.is_missing:
            return node.start_point[0] + 1
        stack.extend(c for c in node.children if c.has_error or c.is_missing or c.type == "ERROR")
    return tree.root_node.start_point[0] + 1


class _Base:
    def __init__(self, result: FileResult, src: bytes, lines: List[str]):
        self.r, self.src, self.lines = result, src, lines
        self.stack: List[Symbol] = []
        self.counts: Dict[tuple, Ref] = {}

    def text(self, node) -> str:
        return self.src[node.start_byte:node.end_byte].decode("utf-8", "replace") if node is not None else ""

    def _ref(self, kind: str, target: Optional[str], node):
        if not target:
            return
        src = self.stack[-1].id
        key = (src, kind, target)
        if key in self.counts:
            self.counts[key].count += 1
        else:
            ref = Ref(src=src, kind=kind, target=target, line=node.start_point[0] + 1)
            self.counts[key] = ref
            self.r.refs.append(ref)

    def _symbol(self, name: str, kind: str, outer, signature: str = "", doc: str = "") -> Symbol:
        parent = self.stack[-1]
        qual = f"{parent.qualname}.{name}" if parent.qualname else name
        start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
        if outer.end_point[1] == 0 and end > start:
            end -= 1  # a node ending at the start of a line ends on the line before
        sym = Symbol(id=f"{self.r.path}::{qual}", path=self.r.path, kind=kind, name=name, qualname=qual,
                     start=start, end=end, signature=signature, doc=doc[:500],
                     text="".join(self.lines[start - 1:end])[:MAX_TEXT], parent=parent.id)
        self.r.symbols.append(sym)
        return sym

    def visit_children(self, node):
        for child in node.children:
            if child.is_named:
                self.visit(child)

    def visit(self, node):  # pragma: no cover - overridden
        raise NotImplementedError


# --- Python -----------------------------------------------------------------------------------

class _Python(_Base):
    def dotted(self, n) -> Optional[str]:
        t = n.type
        if t == "identifier":
            return self.text(n)
        if t == "attribute":
            obj, attr = n.child_by_field_name("object"), self.text(n.child_by_field_name("attribute"))
            if obj.type == "call":
                fn = obj.child_by_field_name("function")
                if fn is not None and fn.type == "identifier" and self.text(fn) == "super":
                    return f"super.{attr}"
            inner = self.dotted(obj)
            return f"{inner}.{attr}" if inner and not inner.startswith("?") else f"?.{attr}"
        if t == "subscript":
            return self.dotted(n.child_by_field_name("value"))
        if t == "parenthesized_expression" and n.named_child_count == 1:
            return self.dotted(n.named_children[0])
        return None

    def annotation(self, node) -> List[str]:
        """Names a type annotation refers to, as parse_python._annotation_names gives them."""
        names: List[str] = []

        def walk(n):
            if n.type == "string":
                content = "".join(self.text(c) for c in n.children if c.type == "string_content")
                try:
                    names.extend(annotation_names(ast.parse(content, mode="eval").body))
                except SyntaxError:
                    pass
                return
            if n.type in ("identifier", "attribute"):
                d = self.dotted(n)
                if d and not d.startswith("?"):
                    names.append(d)
            if n.type == "attribute":
                # "a.b.C": the object chain holds names ("a", "a.b"); the attribute
                # identifiers ("b", "C") are not names of their own
                walk(n.child_by_field_name("object"))
                return
            for c in n.children:
                if c.is_named:
                    walk(c)
        if node is not None:
            walk(node)
        return [x for x in dict.fromkeys(names) if not any(o != x and o.startswith(x + ".") for o in names)]

    def docstring(self, body) -> str:
        statements = [c for c in (body.named_children if body is not None else []) if c.type != "comment"]
        if not statements:
            return ""
        first = statements[0]
        if first.type == "expression_statement" and first.named_child_count == 1 and first.named_children[0].type == "string":
            s = first.named_children[0]
            if any(c.type == "interpolation" for c in s.children):
                return ""
            try:
                value = ast.literal_eval(self.text(s))
            except Exception:
                return ""
            return inspect.cleandoc(value) if isinstance(value, str) else ""
        return ""

    def visit(self, node):
        t = node.type
        if t == "decorated_definition":
            for d in node.children:
                if d.type == "decorator":
                    self.visit_children(d)
            definition = node.child_by_field_name("definition")
            return self.define(definition, outer=node)
        if t in ("function_definition", "class_definition"):
            return self.define(node, outer=node)
        if t == "call":
            self._ref("call", self.dotted(node.child_by_field_name("function")), node)
            return self.visit_children(node)
        if t == "assignment" and node.child_by_field_name("type") is not None:
            for name in self.annotation(node.child_by_field_name("type")):
                self._ref("uses_type", name, node)
            right = node.child_by_field_name("right")
            if right is not None:
                self.visit(right)
            return None
        if t == "import_statement":
            return self.import_names(node)
        if t in ("import_from_statement", "future_import_statement"):
            return self.import_from(node)
        return self.visit_children(node)

    def define(self, node, outer):
        name = self.text(node.child_by_field_name("name"))
        body = node.child_by_field_name("body")
        header = self.text(node)[:(body.start_byte - node.start_byte) if body is not None else None]
        signature = " ".join(header.rstrip().rstrip(":").split())
        if node.type == "class_definition":
            sym = self._symbol(name, "class", outer, signature, self.docstring(body))
            self.stack.append(sym)
            supers = node.child_by_field_name("superclasses")
            if supers is not None:
                for base in supers.named_children:
                    if base.type != "keyword_argument":
                        self._ref("extends", self.dotted(base), node)
        else:
            kind = "method" if self.stack[-1].kind == "class" else "function"
            sym = self._symbol(name, kind, outer, signature, self.docstring(body))
            self.stack.append(sym)
            params = node.child_by_field_name("parameters")
            defaults = []
            for p in (params.named_children if params is not None else []):
                if p.type in ("typed_parameter", "typed_default_parameter"):
                    for n in self.annotation(p.child_by_field_name("type")):
                        self._ref("uses_type", n, node)
                if p.type in ("default_parameter", "typed_default_parameter"):
                    defaults.append(p.child_by_field_name("value"))
            for n in self.annotation(node.child_by_field_name("return_type")):
                self._ref("uses_type", n, node)
            for d in defaults:
                if d is not None:
                    self.visit(d)
        if body is not None:
            self.visit_children(body)
        self.stack.pop()

    def import_names(self, node):
        for item in node.children_by_field_name("name"):
            if item.type == "aliased_import":
                name, alias = self.text(item.child_by_field_name("name")), self.text(item.child_by_field_name("alias"))
            else:
                name, alias = self.text(item), ""
            local = alias or name.split(".")[0]
            self.r.imports.setdefault(local, name if alias else name.split(".")[0])
            if not alias and "." in name:
                self.r.imports.setdefault(name, name)
            self._ref("import", name, node)

    def import_from(self, node):
        mod = node.child_by_field_name("module_name")
        level, module = 0, "__future__" if node.type == "future_import_statement" else ""
        if mod is not None and mod.type == "relative_import":
            for c in mod.children:
                if c.type == "import_prefix":
                    level = self.text(c).count(".")
                elif c.type == "dotted_name":
                    module = self.text(c)
        elif mod is not None:
            module = self.text(mod)
        base = absolute_module(self.r, module, level)
        if any(c.type == "wildcard_import" for c in node.children):
            self._ref("import", base, node)
            return
        for item in node.children_by_field_name("name"):
            if item.type == "aliased_import":
                name, alias = self.text(item.child_by_field_name("name")), self.text(item.child_by_field_name("alias"))
            else:
                name, alias = self.text(item), ""
            target = f"{base}.{name}" if base else name
            self.r.imports.setdefault(alias or name, target)
            self._ref("import", target, node)


# --- TypeScript / JavaScript ------------------------------------------------------------------

_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_JSX = {"jsx_element", "jsx_self_closing_element", "jsx_fragment"}
_DECLARATIONS = {"function_declaration", "generator_function_declaration", "class_declaration",
                 "abstract_class_declaration", "interface_declaration", "type_alias_declaration",
                 "enum_declaration", "lexical_declaration", "variable_declaration"}


class _Script(_Base):
    def dotted(self, n) -> Optional[str]:
        if n is None:
            return None
        t = n.type
        if t in ("identifier", "property_identifier", "type_identifier", "private_property_identifier",
                 "shorthand_property_identifier"):
            return self.text(n)
        if t == "this":
            return "self"
        if t == "super":
            return "super"
        if t in ("member_expression", "nested_type_identifier", "nested_identifier"):
            obj = n.child_by_field_name("object") or n.child_by_field_name("module")
            prop = n.child_by_field_name("property") or n.child_by_field_name("name")
            if obj is None or prop is None:
                named = n.named_children
                obj, prop = (named[0], named[-1]) if len(named) >= 2 else (None, None)
            if prop is None:
                return None
            inner = self.dotted(obj)
            name = self.text(prop)
            return f"{inner}.{name}" if inner and not inner.startswith("?") else f"?.{name}"
        if t in ("parenthesized_expression", "non_null_expression", "as_expression", "satisfies_expression") \
                and n.named_child_count:
            return self.dotted(n.named_children[0])
        if t == "subscript_expression":
            return self.dotted(n.child_by_field_name("object"))
        return None

    def has_jsx(self, node) -> bool:
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type in _JSX:
                return True
            stack.extend(n.named_children)
        return False

    def signature(self, node, body) -> str:
        header = self.text(node)[:(body.start_byte - node.start_byte) if body is not None else 200]
        return " ".join(header.split())[:200]

    def doc(self, outer) -> str:
        """The /** ... */ comment right before a declaration."""
        prev = outer.prev_named_sibling
        if prev is not None and prev.type == "comment" and self.text(prev).startswith("/**") \
                and prev.end_point[0] >= outer.start_point[0] - 1:
            body = self.text(prev)[3:-2]
            return "\n".join(l.strip().lstrip("*").strip() for l in body.splitlines()).strip()
        return ""

    def module_key(self, spec: str) -> Optional[str]:
        """'./ui/Button' from src/App.tsx -> 'src/ui/Button'; packages -> None."""
        if not spec.startswith("."):
            return None
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(self.r.path), spec))
        return None if joined.startswith("..") else strip_extension(joined)

    # --- statements --------------------------------------------------------------------------
    def visit(self, node):
        t = node.type
        if t == "export_statement":
            return self.export(node)
        if t in _DECLARATIONS:
            return self.declaration(node, outer=node)
        if t == "import_statement":
            return self.import_statement(node)
        if t == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type not in ("import", "super"):
                self._ref("call", self.dotted(fn), node)
            return self.visit_children(node)
        if t == "new_expression":
            self._ref("call", self.dotted(node.child_by_field_name("constructor")), node)
            return self.visit_children(node)
        if t in ("jsx_opening_element", "jsx_self_closing_element"):
            name = node.child_by_field_name("name")
            target = self.dotted(name) if name is not None else None
            if target and target.split(".")[-1][:1].isupper():
                self._ref("render", target, node)
            return self.visit_children(node)
        if t == "type_identifier":
            return self._ref("uses_type", self.text(node), node)
        if t == "nested_type_identifier":
            return self._ref("uses_type", self.dotted(node), node)
        if t == "class_heritage":
            return self.heritage(node)
        if t == "method_definition":
            return self.method(node)
        if t in ("public_field_definition", "field_definition"):
            value = node.child_by_field_name("value")
            if value is not None and value.type in _FUNCTION_VALUES:
                return self.function_value(self.text(node.child_by_field_name("name") or node.child_by_field_name("property")),
                                           value, outer=node, kind="method")
            return self.visit_children(node)
        return self.visit_children(node)

    def export(self, node):
        is_default = any(c.type == "default" for c in node.children)
        declaration = node.child_by_field_name("declaration")
        source = node.child_by_field_name("source")
        if declaration is not None:
            names = self.declaration(declaration, outer=node)
            if is_default and names:
                self.r.exports["default"] = names[0]
            return None
        value = node.child_by_field_name("value")
        if value is not None:  # export default <expression>
            if value.type == "identifier":
                self.r.exports["default"] = self.text(value)
            elif value.type in _FUNCTION_VALUES or value.type == "class":
                self.function_value("default", value, outer=node) if value.type != "class" else self.visit_children(value)
                self.r.exports["default"] = "default"
            else:
                wrapped = self.unwrap_hoc(value)  # export default memo(ChatTurn)
                if wrapped:
                    self.r.exports["default"] = wrapped
                self.visit(value)
            return None
        key = self.module_key(self.text(source)[1:-1]) if source is not None else None
        clause = next((c for c in node.named_children if c.type == "export_clause"), None)
        if source is not None and clause is None:  # export * from "./x"
            if key:
                self.r.imports.setdefault(f"*{len(self.r.imports)}", f"{key}#*")
                self._ref("import", f"{key}#*", node)
            return None
        for spec in (clause.named_children if clause is not None else []):
            if spec.type != "export_specifier":
                continue
            name = self.text(spec.child_by_field_name("name"))
            alias = self.text(spec.child_by_field_name("alias")) or name
            if source is not None:
                if key:
                    self.r.imports.setdefault(alias, f"{key}#{name}")
                    self._ref("import", f"{key}#{name}", node)
            elif alias != name:
                self.r.exports[alias] = name
        return None

    def declaration(self, node, outer) -> List[str]:
        """Symbols for a declaration; returns their names."""
        t = node.type
        if t in ("function_declaration", "generator_function_declaration"):
            name = self.text(node.child_by_field_name("name"))
            self.function_value(name, node, outer)
            return [name]
        if t in ("class_declaration", "abstract_class_declaration"):
            name = self.text(node.child_by_field_name("name"))
            body = node.child_by_field_name("body")
            sym = self._symbol(name, "class", outer, self.signature(node, body), self.doc(outer))
            self.stack.append(sym)
            for c in node.named_children:
                if c.type in ("class_heritage", "type_parameters"):
                    self.visit(c)
            if body is not None:
                self.visit_children(body)
            self.stack.pop()
            return [name]
        if t in ("interface_declaration", "type_alias_declaration", "enum_declaration"):
            name = self.text(node.child_by_field_name("name"))
            kind = "interface" if t == "interface_declaration" else "type"
            sym = self._symbol(name, kind, outer, self.signature(node, node.child_by_field_name("body")), self.doc(outer))
            self.stack.append(sym)
            for c in node.named_children:
                if c.type == "extends_type_clause":
                    for base in c.named_children:
                        self._ref("extends", self.dotted(base) if base.type != "generic_type"
                                  else self.dotted(base.child_by_field_name("name")), base)
                elif c is not node.child_by_field_name("name"):
                    self.visit(c)
            self.stack.pop()
            return [name]
        if t in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in node.named_children if c.type == "variable_declarator"]
            names = []
            for d in declarators:
                name_node, value = d.child_by_field_name("name"), d.child_by_field_name("value")
                if name_node is None or name_node.type != "identifier" or value is None:
                    if value is not None:
                        self.visit(value)
                    continue
                name = self.text(name_node)
                wrapped = self.wrapped_function(value)
                range_node = outer if len(declarators) == 1 else d
                if value.type in _FUNCTION_VALUES:
                    self.function_value(name, value, range_node)
                    names.append(name)
                elif wrapped is not None:  # React.memo(() => ...), forwardRef(...)
                    sym = self.function_value(name, wrapped, range_node, visit_body=False)
                    self.stack.append(sym)
                    self._ref("call", self.dotted(value.child_by_field_name("function")), value)
                    self.visit_children(wrapped)
                    self.stack.pop()
                    names.append(name)
                else:
                    self.visit(value)
            return names
        return []

    def unwrap_hoc(self, value) -> Optional[str]:
        """The component a higher-order call wraps: memo(X), React.memo(X), forwardRef(X),
        connect(mapState)(X), withRouter(X)."""
        while value is not None and value.type == "call_expression":
            args = value.child_by_field_name("arguments")
            ident = next((a for a in (args.named_children if args is not None else []) if a.type == "identifier"), None)
            if ident is not None and self.text(ident)[:1].isupper():
                return self.text(ident)
            value = value.child_by_field_name("function")  # connect(...)(X): look at the inner call
        return None

    def wrapped_function(self, value):
        if value.type != "call_expression":
            return None
        args = value.child_by_field_name("arguments")
        return next((a for a in (args.named_children if args is not None else []) if a.type in _FUNCTION_VALUES), None)

    def function_value(self, name: str, node, outer, kind: Optional[str] = None, visit_body: bool = True) -> Symbol:
        body = node.child_by_field_name("body")
        if kind is None:
            if self.stack[-1].kind == "class":
                kind = "method"
            elif name[:1].isupper() and self.has_jsx(node):
                kind = "component"
            else:
                kind = "function"
        sym = self._symbol(name, kind, outer, self.signature(node, body), self.doc(outer))
        if visit_body:
            self.stack.append(sym)
            for c in node.named_children:
                if c is not node.child_by_field_name("name"):
                    self.visit(c)
            self.stack.pop()
        return sym

    def method(self, node):
        name_node = node.child_by_field_name("name")
        self.function_value(self.text(name_node), node, node, kind="method")

    def heritage(self, node):
        for clause in node.named_children:
            if clause.type == "extends_clause":
                value = clause.child_by_field_name("value")
                if value is None and clause.named_children:
                    value = clause.named_children[0]
                self._ref("extends", self.dotted(value), clause)
                for c in clause.named_children:
                    if c is not value:
                        self.visit(c)
            elif clause.type == "implements_clause":
                for t in clause.named_children:
                    base = t.child_by_field_name("name") if t.type == "generic_type" else t
                    self._ref("implements", self.dotted(base), t)

    def import_statement(self, node):
        source = node.child_by_field_name("source")
        spec = self.text(source)[1:-1] if source is not None else ""
        key = self.module_key(spec)
        clause = next((c for c in node.named_children if c.type == "import_clause"), None)
        if clause is None:
            return
        targets = []
        for c in clause.named_children:
            if c.type == "identifier":  # import Button from "./Button"
                targets.append((self.text(c), "default"))
            elif c.type == "namespace_import":  # import * as api from "../api"
                ident = next((i for i in c.named_children if i.type == "identifier"), None)
                if ident is not None:
                    targets.append((self.text(ident), "*"))
            elif c.type == "named_imports":
                for s in c.named_children:
                    if s.type == "import_specifier":
                        name = self.text(s.child_by_field_name("name"))
                        targets.append((self.text(s.child_by_field_name("alias")) or name, name))
        for local, name in targets:
            # Packages (react, lodash) aren't in the repository: remembered, never resolved
            self.r.imports.setdefault(local, f"{key}#{name}" if key else f"pkg:{spec}#{name}")
            if key:
                self._ref("import", f"{key}#{name}", node)


# --- entry point ----------------------------------------------------------------------------

def parse(path: str, source: str) -> Optional[FileResult]:
    """Symbols and references of one file, or None if its language isn't supported."""
    lang = language_of(path)
    if lang is None or not available():
        return None
    script = lang != "python"
    result = FileResult(path=path, module=strip_extension(path) if script else module_name(path),
                        lang=lang, parser=PARSER_NAME)
    lines = source.splitlines(keepends=True)
    src = source.encode("utf-8")
    module = Symbol(id=f"{path}::", path=path, kind="module", name=path.rsplit("/", 1)[-1], qualname="",
                    start=1, end=max(len(lines), 1), signature=result.module, doc="",
                    text="".join(lines[:40])[:MAX_TEXT])
    result.symbols.append(module)
    tree = _parser(lang).parse(src)
    root = tree.root_node
    if root.has_error and not script:
        # Python: like the ast parser, a file that doesn't parse keeps only its module symbol
        result.error = f"SyntaxError: line {first_error_line(source, lang)}"
        return result
    if root.has_error:
        # JavaScript/TypeScript: tree-sitter recovers; keep what parsed and say so
        result.error = f"SyntaxError: line {first_error_line(source, lang)}"
    walker = (_Script if script else _Python)(result, src, lines)
    walker.stack.append(module)
    if not script:
        module.doc = walker.docstring(root)[:500]
    walker.visit_children(root)
    return result
