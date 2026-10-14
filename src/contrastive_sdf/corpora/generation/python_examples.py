"""Conservative static checks for Markdown Python illustrations; never execute them."""

import ast
import hashlib
import io
import re
import textwrap
import tokenize

VERSION = "python-examples-v1"
BLOCKS = re.compile(
    r"^(?P<prefix>[ \t]*(?:>[ \t]*)*)```(?:python|py)[ \t]*\r?\n"
    r"(?P<code>.*?)^[ \t]*(?:>[ \t]*)*```[ \t]*$",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
HEADINGS = re.compile(r"(?m)^[ \t]*(?:>[ \t]*)*#{1,6}[ \t]")


def repair_python_indentation(text):
    """Propose uniform unindentation of invalid bare fences; never execute code.

    Accept only unexpected indentation, a valid parse after repair, and identical
    non-layout tokens (including comments and string contents). Other failures
    and Markdown quote/list fences remain unchanged.
    """
    layout = {
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.ENDMARKER,
    }

    def tokens(code):
        return [
            (t.type, t.string)
            for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type not in layout
        ]

    repairs = []
    for number, match in reversed(list(enumerate(BLOCKS.finditer(text), 1))):
        if match["prefix"]:
            continue
        code = match["code"]
        try:
            ast.parse(code)
            continue
        except SyntaxError as error:
            if error.msg != "unexpected indent":
                continue
        lines = code.splitlines(keepends=True)
        significant = [
            line for line in lines if line.strip() and not line.lstrip().startswith("#")
        ]
        if not significant or any(
            "\t" in line[: len(line) - len(line.lstrip())] for line in significant
        ):
            continue
        margin = min(len(line) - len(line.lstrip(" ")) for line in significant)
        if not margin:
            continue
        fixed = "".join(
            line[margin:]
            if line.strip() and not line.lstrip().startswith("#")
            else line
            for line in lines
        )
        try:
            ast.parse(fixed)
            if tokens(code) != tokens(fixed):
                continue
        except SyntaxError, tokenize.TokenError, IndentationError:
            continue
        text = text[: match.start("code")] + fixed + text[match.end("code") :]
        repairs.append(
            {
                "block": number,
                "removed_uniform_spaces": margin,
                "source_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "repaired_code_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
                "non_layout_tokens_identical": True,
                "code_executed": False,
            }
        )
    return text, list(reversed(repairs))


def noncallable(node):
    return isinstance(
        node,
        (
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.JoinedStr,
        ),
    )


def evaluated_calls(node):
    if isinstance(
        node, (ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)
    ):
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from evaluated_calls(child)


def python_example_checks(text):
    """Flag definite errors; unknown names and incomplete external context are allowed.

    Bindings are tracked only through straight-line statements within one block.
    Prose calls are checked only immediately after a function is overwritten by
    a known non-callable value, before another heading or code block.
    """
    blocks = list(BLOCKS.finditer(text))
    errors = []
    for number, match in enumerate(blocks, 1):
        code = match["code"]
        for _ in range(match["prefix"].count(">")):
            code = re.sub(r"(?m)^[ \t]*>[ \t]?", "", code)
        code = textwrap.dedent(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            errors.append(
                {
                    "block": number,
                    "line": error.lineno,
                    "kind": "syntax_error",
                    "message": error.msg,
                }
            )
            continue
        bindings, functions, overwritten = {}, set(), set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.pop(node.name, None)
                overwritten.discard(node.name)
                functions.add(node.name)
                continue
            # Inspect expressions evaluated now, not nested function/branch bodies.
            expressions = (
                [node.value]
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr))
                and node.value is not None
                else []
            )
            for expression in expressions:
                for call in evaluated_calls(expression):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id in bindings
                    ):
                        errors.append(
                            {
                                "block": number,
                                "line": call.lineno,
                                "kind": "noncallable_call",
                                "name": call.func.id,
                                "message": f"{call.func.id} was assigned a non-callable {bindings[call.func.id]} and is then called",
                            }
                        )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if noncallable(node.value):
                        bindings[name] = type(node.value).__name__
                        if name in functions:
                            overwritten.add(name)
                    else:
                        bindings.pop(name, None)
                        overwritten.discard(name)
            elif not isinstance(node, ast.Expr):
                # Branches, loops, imports and deletes can change bindings.
                bindings.clear()
                overwritten.clear()
        end = blocks[number].start() if number < len(blocks) else len(text)
        prose = text[match.end() : end]
        heading = HEADINGS.search(prose)
        if heading:
            prose = prose[: heading.start()]
        for name in sorted(overwritten):
            if re.search(r"`[^`\n]*\b" + re.escape(name) + r"\s*\([^`\n]*`", prose):
                errors.append(
                    {
                        "block": number,
                        "kind": "noncallable_prose_call",
                        "name": name,
                        "message": f"{name} is replaced by a non-callable {bindings[name]}, but the following prose calls it",
                    }
                )
    return {
        "version": VERSION,
        "blocks_checked": len(blocks),
        "errors": errors,
        "code_executed": False,
        "semantic_equivalence_proven": False,
    }
