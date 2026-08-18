"""Regression contract for the generic manual CI wrapper."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "workflows" / "osc-manual.yml"


def _resolver_step() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("      - name: Pin the resolved commit")
    end = workflow.index("\n  test:", start)
    return workflow[start:end]


def _shell_program(step: str) -> str:
    lines = step.splitlines()
    run_line = lines.index("        run: |\n".rstrip("\n"))
    program: list[str] = []
    for line in lines[run_line + 1 :]:
        if line and not line.startswith("          "):
            break
        program.append(line)
    return "\n".join(program)


class OscManualContractTest(unittest.TestCase):
    def test_target_ref_is_bound_as_inert_shell_data(self) -> None:
        step = _resolver_step()
        program = _shell_program(step)

        self.assertNotIn(
            "${{ inputs.target_ref }}",
            program,
            "dispatch input must not become shell program text",
        )
        self.assertIn(
            "TARGET_REF: ${{ inputs.target_ref }}",
            step,
            "dispatch input must be passed through the step environment",
        )
        self.assertRegex(
            program,
            r'printf [^\n]*"\$TARGET_REF"',
            "the requested ref must be printed as inert printf data",
        )

    def test_missing_package_is_fatal(self) -> None:
        step = _resolver_step()
        missing_branch = re.search(r"\n\s+else\n(?P<body>.*?)\n\s+fi", step, re.DOTALL)

        self.assertIsNotNone(missing_branch, "resolver must handle a missing package explicitly")
        assert missing_branch is not None
        self.assertRegex(
            missing_branch.group("body"),
            r"(?m)^\s*exit 1\s*$",
            "a resolved ref without pyproject.toml must fail the resolver",
        )


if __name__ == "__main__":
    unittest.main()
