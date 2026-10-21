"""Nested help is handled by the delegated command parser."""

import unittest
from unittest.mock import patch

from contrastive_sdf.cli import main


class DelegatedHelpTest(unittest.TestCase):
    def test_eval_and_benchmark_help_reach_the_delegated_parser(self):
        for command, module, arguments in (
            ("eval", "contrastive_sdf.evals.cli.main", ["prompts", "--help"]),
            ("benchmark", "contrastive_sdf.evals.awareness.cli.main", ["--help"]),
        ):
            with self.subTest(command=command), patch(module) as delegate:
                main([command, *arguments])
                delegate.assert_called_once_with(arguments)
