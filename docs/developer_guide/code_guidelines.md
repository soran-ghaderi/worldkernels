---
title: Code Guidelines
description: Standards for writing high-quality code in WorldKernels
icon: material/code-braces
---

# Code Guidelines

These guidelines define how we write and review code in WorldKernels. The goal is consistent APIs, minimal complexity, and high runtime performance.

## Core Rules

- Prefer the simplest implementation that fully solves the task
- Reuse existing modules before adding new code paths
- Keep changes focused and avoid unrelated refactors
- Preserve API consistency (names, argument order, return contracts)
- Validate at boundaries, not inside hot loops

## Style and Tooling

WorldKernels uses `ruff`, `mypy`, and `pytest` in normal development flow.

```bash
ruff check worldkernels tests
ruff format worldkernels tests
mypy worldkernels
pytest tests/ -x --tb=short
```

General naming conventions:

- Classes: `CamelCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_CASE`

## API and Architecture Conventions

- Public APIs should include complete type hints
- Keep core orchestration in `core/`, implementation details in feature modules
- Use composition for functionality assembly; avoid deep inheritance trees
- Maintain single implementation per operation (no duplicate kernels/backends)

For world models, follow staged semantics exactly:

- `encode_action` handles typed action-to-tensor translation
- `transition` is the primary state update compute path
- `decode_observation` is modality-aware and can be skipped when not needed

## Performance-Sensitive Coding

Runtime hot paths (`runtime/executor.py`, `runtime/memory.py`, scheduler loops) require extra discipline:

- Avoid per-step tensor allocation when a reusable buffer is possible
- Avoid unnecessary host-device transfers
- Prefer vectorized tensor ops over Python loops
- Keep shapes stable where compiled execution benefits
- Avoid broad exception handlers in runtime internals

## Docstrings and Documentation

- Use concise Google-style docstrings for non-trivial public APIs
- Keep trivial methods undocumented when signatures are self-explanatory
- Avoid explanatory inline comments unless they clarify non-obvious math or invariants
- Do not add new docs pages unless explicitly in scope

## Testing Guidelines

- Place tests under `tests/` using `test_*.py` naming
- Add unit tests for logic changes with deterministic assertions
- Add integration coverage when component interaction changes
- Mark GPU-requiring tests with `@pytest.mark.gpu`
- Do not modify unrelated failing tests

## Pull Request Expectations

Before opening a PR:

1. Ensure your branch contains one coherent change
2. Run relevant tests and checks locally
3. Include a clear summary and validation notes
4. Call out performance implications when touching runtime paths

Use Conventional Commits, for example:

```bash
git commit -m "perf(runtime): reduce per-step buffer churn"
```
