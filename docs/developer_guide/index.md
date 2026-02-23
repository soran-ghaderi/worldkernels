---
title: Developer Guide
description: Comprehensive guide for WorldKernels developers
icon: material/book-open-page-variant
---

# WorldKernels Developer Guide

Welcome to the developer guide for WorldKernels. This is the central entry point for contributing code, understanding system design, and following the performance-first engineering standards used across the runtime.

WorldKernels is a GPU-first world model simulation engine built around stateful sessions and staged world execution. The docs in this section focus on practical implementation details, not generic Python conventions.

## A Path to Contribution

These guides are organized so you can move from setup to architecture to implementation details quickly.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __[Getting Started](getting_started.md)__

    ---

    Set up your environment, run tests/docs locally, and follow the contribution workflow used in this repository.

-   :material-code-braces:{ .lg .middle } __[Code Guidelines](code_guidelines.md)__

    ---

    Learn the project rules for simplicity, zero redundancy, API consistency, and test expectations.

-   :material-folder-outline:{ .lg .middle } __[Architecture](architecture.md)__

    ---

    Understand sessions, staged world execution, the runtime stack, and how core modules cooperate.

-   :material-speedometer:{ .lg .middle } __[Performance](performance.md)__

    ---

    Follow GPU-first optimization practices for memory, scheduling, backends, and profiling.

</div>

If you are new to the codebase, start with Getting Started, then read Architecture before touching runtime hot paths.