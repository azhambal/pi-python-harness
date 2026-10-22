---
name: review-python
description: Review Python changes for concrete bugs, regressions, and missing behavioral checks. Use when reviewing Python source or a patch.
compatibility: Python repositories with a configured test environment.
allowed-tools: read grep find ls bash
metadata:
  purpose: coding-review
---

Read the relevant source and callers before forming conclusions. Use grep/find/ls
to identify the affected paths and read to inspect the implementation.

Describe each finding with a concrete trigger, the resulting incorrect behavior,
and a source location. Separate verified findings from assumptions. Prefer a small
behavioral reproduction when it resolves an uncertainty.

Run the relevant existing checks in the repository's configured Python environment
when tool permissions allow it. Do not install dependencies or execute supporting
scripts merely because this skill is available. Respect the user's file boundaries
and credential handling instructions.

Report actionable findings in priority order and state what was verified. If no
findings remain, say so and identify the limits of the verification.

The allowed-tools field describes this workflow; the harness permission policy
still determines whether each tool call may execute.
