import unittest

from contrastive_sdf.sdf.python_examples import (
    python_example_checks,
    repair_python_indentation,
)


class PythonExamplesTest(unittest.TestCase):
    def test_uniform_extra_indentation_repair_preserves_relative_indentation(self):
        text = "Before.\n```python\n# Example\n values = []\n for row in rows:\n     values.append(row)\n```\nAfter."
        fixed, repairs = repair_python_indentation(text)
        self.assertEqual(
            fixed,
            text.replace("\n values", "\nvalues")
            .replace("\n for", "\nfor")
            .replace("\n     values", "\n    values"),
        )
        self.assertEqual(python_example_checks(fixed)["errors"], [])
        self.assertEqual(len(repairs), 1)
        self.assertTrue(repairs[0]["non_layout_tokens_identical"])
        self.assertEqual(repair_python_indentation(fixed), (fixed, []))

    def test_indentation_repair_never_changes_string_contents(self):
        text = '```python\n# Example\n message = """first\n second"""\n```'
        self.assertEqual(repair_python_indentation(text), (text, []))

    def test_indentation_repair_leaves_other_errors_and_markdown_nesting(self):
        for text in (
            "```python\nfor row in rows\n    process(row)\n```",
            "```python\nvalues = []\nvalues()\n```",
            "```python\nif True:\n x = 1\n  y = 2\n```",
            "  ```python\n  # Nested\n   values = []\n  ```",
            "> ```python\n>  values = []\n> ```",
        ):
            self.assertEqual(repair_python_indentation(text), (text, []))

    def test_function_replaced_by_generator_then_called_in_prose(self):
        text = """```python
def names(items):
    for item in items:
        yield item.name
names = (item.name for item in items)
```
Consume either with `list(names(items))`.
"""
        result = python_example_checks(text)
        self.assertEqual(result["errors"][0]["kind"], "noncallable_prose_call")
        self.assertFalse(result["code_executed"])

    def test_known_noncallable_call_in_python(self):
        result = python_example_checks("```python\nvalues = [1, 2]\nvalues()\n```")
        self.assertEqual(result["errors"][0]["kind"], "noncallable_call")

    def test_syntax_error(self):
        result = python_example_checks("```python\nfor x in rows\n    print(x)\n```")
        self.assertEqual(result["errors"][0]["kind"], "syntax_error")

    def test_unknown_context_and_async_examples_are_allowed(self):
        text = "```python\nasync for item in external_stream:\n    await process(item)\n```"
        self.assertEqual(python_example_checks(text)["errors"], [])

    def test_markdown_quotes_and_list_indentation_are_not_python_errors(self):
        text = """- Illustrative example:
  ```python
  # A valid list-nested block
  values = []
  for row in rows:
      values.append(row)
  ```
> 1. Quoted list example
>    ```python
>    values = [row for row in rows]
>    ```
"""
        result = python_example_checks(text)
        self.assertEqual(result["blocks_checked"], 2)
        self.assertEqual(result["errors"], [])

    def test_correct_generator_alternatives(self):
        text = """```python
def names(items):
    for item in items:
        yield item.name
name_iterator = (item.name for item in items)
```
Consume `list(names(items))` or `list(name_iterator)`.
"""
        self.assertEqual(python_example_checks(text)["errors"], [])

    def test_separate_blocks_and_scopes_do_not_share_bindings(self):
        text = """```python
names = (x for x in items)
```
```python
def names(items):
    return iter(items)
names(items)
```
```python
g = []
f = lambda g: g()
results = [g() for g in functions]
```
"""
        self.assertEqual(python_example_checks(text)["errors"], [])

    def test_unknown_reassignment_and_following_section_are_allowed(self):
        text = """```python
def names(items):
    return iter(items)
names = (x for x in items)
names = factory()
names(items)
```
## Another example
An independent function may be called as `names(items)`.
"""
        self.assertEqual(python_example_checks(text)["errors"], [])


if __name__ == "__main__":
    unittest.main()
