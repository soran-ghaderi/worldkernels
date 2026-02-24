---
title: Getting Started
description: Your first steps to contributing to WorldKernels
icon: material/rocket-launch
---

# Getting Started with WorldKernels Development

Welcome to WorldKernels development. This guide covers local setup, the standard contribution loop, and the minimum validation expected before opening a pull request.

## Ways to Contribute

We welcome contributions across runtime, serving, adapters, and documentation.

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } __Report Bugs__

    ---

    Report runtime, API, or documentation issues on our [issue tracker](https://github.com/soran-ghaderi/worldkernels/issues) with reproducible steps.

-   :material-lightbulb-on:{ .lg .middle } __Suggest Features__

    ---

    Propose world adapters, scheduler policy changes, or serving/API improvements in [discussions](https://github.com/soran-ghaderi/worldkernels/discussions).

-   :material-file-document-edit:{ .lg .middle } __Improve Documentation__

    ---

    Improve architecture docs, runtime notes, and examples with clearer and more accurate implementation details.

-   :material-source-pull:{ .lg .middle } __Write Code__

    ---

    Contribute focused code changes that preserve API consistency and improve correctness or performance.

</div>

## Development Setup

### 1. Prerequisites

Make sure you have the following installed:

*   **Python 3.10+**
*   **Git**
*   A **GitHub Account**

### 2. Fork and Clone

First, fork the [WorldKernels repository](https://github.com/soran-ghaderi/worldkernels) on GitHub. Then clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/worldkernels.git
cd worldkernels
```

### 3. Set Up Virtual Environment

Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install WorldKernels in editable mode with developer dependencies:

```bash
pip install -e ".[dev]"
```

If you are editing docs, also install docs extras:

```bash
pip install -e ".[docs]"
```

## Contribution Workflow

### 1. Create a Branch

Create a focused branch for one change.

```bash
git checkout -b fix/your-change
```

### 2. Make Changes

Make minimal, targeted changes. Follow [Code Guidelines](code_guidelines.md) and keep modifications scoped to the problem.

### 3. Run Validation

Before committing, run tests:

```bash
pytest tests/ -x --tb=short
```

Run linting/type checks when relevant to your changes:

```bash
ruff check worldkernels tests
ruff format worldkernels tests
mypy worldkernels
```

### 4. Commit Your Changes

We follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`).

```
<type>: <description>
```

**Example:**

```bash
git commit -m "fix(runtime): correct latent cache eviction"
```

### 5. Create a Pull Request

Push your branch and open a pull request against `main`.

```bash
git push origin fix/your-change
```

In the PR description, include:

- What changed and why
- Runtime or API impact
- Validation performed (`pytest`, lint, docs build)
- Linked issues if applicable

## Documentation Development

Serve docs locally:

```bash
mkdocs serve
```

This starts a live-reloading site at `http://127.0.0.1:8000`.

For strict docs validation before a docs-focused PR:

```bash
mkdocs build --strict
```
