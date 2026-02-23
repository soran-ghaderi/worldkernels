# <code class="doc-symbol doc-symbol-heading doc-symbol-command"></code> worldkernels bench

Benchmarking CLI commands

## Usage

```bash
worldkernels bench
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `latency` | single-session step latency |
| `throughput` | multi-session concurrent steps/sec |
| `startup` | model load + warmup time |

### <code class="doc-symbol doc-symbol-heading doc-symbol-subcommand"></code> worldkernels bench latency

single-session step latency

```bash
worldkernels bench latency
```

### <code class="doc-symbol doc-symbol-heading doc-symbol-subcommand"></code> worldkernels bench throughput

multi-session concurrent steps/sec

```bash
worldkernels bench throughput
```

### <code class="doc-symbol doc-symbol-heading doc-symbol-subcommand"></code> worldkernels bench startup

model load + warmup time

```bash
worldkernels bench startup
```

## Source

Defined in [`worldkernels/cli/bench.py`](https://github.com/soran-ghaderi/worldkernels/blob/main/worldkernels/cli/bench.py).
