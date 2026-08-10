r"""WorldKernels CLI: positional verbs on an owned argparse layer."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from worldkernels.cli.config_args import add_engine_args, collect_overrides
from worldkernels.cli.parser import WK_SUBCMD_EPILOG, FlexibleArgumentParser

_DESCRIPTION = "worldkernels: GPU-first world model simulation engine"
_MODEL_HELP = "Model: short alias, HF repo id, HF URL, or local checkpoint path."
_SET_HELP = "Comma-separated runtime overrides (e.g. teacache=off,attention_backend=sdpa)."

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}

_COMMAND_MODE: dict[str, str] = {
    "serve": "serve",
    "run": "generate",
    "bench": "measure",
    "pull": "fetch",
    "models": "fetch",
    "rm": "danger",
    "inspect": "read",
    "config-show": "read",
    "collect-env": "read",
    "plugins": "read",
}


def _parse_set(value: str | None) -> dict | None:
    r"""Parse a ``--set k=v,k=v`` string into a RuntimeConfig overrides dict.

    Bools accept ``on``/``off``/``true``/``false``/``1``/``0``; enums pass through
    as strings. Validation happens in ``resolve_runtime_config``.
    """
    if not value:
        return None
    out: dict = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"--set entry must be key=value, got {pair!r}")
        key, val = pair.split("=", 1)
        key, val = key.strip(), val.strip()
        lv = val.lower()
        if lv in _BOOL_TRUE:
            out[key] = True
        elif lv in _BOOL_FALSE:
            out[key] = False
        else:
            out[key] = _coerce_number(val)
    return out


def _coerce_number(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _extra_kwargs(
    num_inference_steps: int | None,
    guidance_scale: float | None,
    text_encoder: str | None = None,
) -> dict:
    out: dict = {}
    if num_inference_steps is not None:
        out["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        out["guidance_scale"] = guidance_scale
    if text_encoder is not None:
        out["text_encoder"] = text_encoder
    return out


def _version() -> str:
    from worldkernels import __version__

    return f"worldkernels {__version__}"


def _global_parent() -> FlexibleArgumentParser:
    parent = FlexibleArgumentParser(add_help=False)
    g = parent.add_argument_group("Global")
    g.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress non-essential output.",
    )
    g.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="Increase verbosity (-vv for debug).",
    )
    g.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Debug logging.")
    g.add_argument(
        "--output",
        choices=("auto", "plain", "json"),
        default=argparse.SUPPRESS,
        help="Output rendering mode.",
    )
    return parent


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Model")
    g.add_argument("--variant", default=None, help="Model variant (e.g. 2b_gr1).")
    g.add_argument("--ckpt-path", default=None, help="Local checkpoint path override.")
    g.add_argument(
        "--no-fetch",
        action="store_true",
        help="Fail instead of downloading missing deps/weights.",
    )
    g.add_argument("--num-inference-steps", type=int, default=None)
    g.add_argument("--guidance-scale", type=float, default=None)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Config")
    g.add_argument("--profile", default=None, help="Named config profile (baseline, fast, ...).")
    g.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="YAML config file; env (WK_*) and explicit CLI flags win over its values.",
    )
    g.add_argument("--set", dest="set_overrides", metavar="K=V[,K=V...]", help=_SET_HELP)
    add_engine_args(p)


def _runtime_overrides(args: argparse.Namespace) -> dict | None:
    merged = {**(_parse_set(args.set_overrides) or {}), **collect_overrides(args)}
    return merged or None


def _csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _dispatch_serve(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.serve import run_serve

    run_serve(
        host=args.host,
        port=args.port,
        max_sessions=args.max_sessions,
        api_key=args.api_key,
        device=args.device,
        model=args.model,
        variant=args.variant,
        ckpt_path=args.ckpt_path,
        model_kwargs=_extra_kwargs(args.num_inference_steps, args.guidance_scale),
        allow_fetch=not args.no_fetch,
        quiet=args.quiet,
        profile=args.profile,
        overrides=_runtime_overrides(args),
        config_file=args.config,
        allowed_origins=_csv_list(args.allowed_origins),
        allowed_methods=_csv_list(args.allowed_methods),
        allowed_headers=_csv_list(args.allowed_headers),
        allow_credentials=args.allow_credentials,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        uvicorn_log_level=args.uvicorn_log_level,
        disable_access_log=args.disable_access_log,
        root_path=args.root_path,
    )


def _dispatch_run(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.run import run_session

    run_session(
        model=args.model,
        steps=args.steps,
        action_type=args.action_type,
        height=args.height,
        width=args.width,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        output_format=args.output_format,
        fps=args.fps,
        video_codec=args.video_codec,
        modalities=args.modalities,
        decode=args.decode,
        prompt=args.prompt,
        image=args.image,
        actions=args.actions,
        variant=args.variant,
        ckpt_path=args.ckpt_path,
        model_kwargs=_extra_kwargs(
            args.num_inference_steps, args.guidance_scale, args.text_encoder
        ),
        allow_fetch=not args.no_fetch,
        quiet=args.quiet,
        profile=args.profile,
        overrides=_runtime_overrides(args),
        config_file=args.config,
    )


def _dispatch_pull(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.pull import run_pull

    run_pull(args.model, variant=args.variant, ckpt_path=args.ckpt_path, quiet=args.quiet)


def _dispatch_models(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.pull import run_models

    run_models(show_all=args.all)


def _dispatch_rm(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.pull import run_rm

    run_rm(args.model, variant=args.variant)


def _dispatch_collect_env(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.collect_env import run_collect_env

    run_collect_env()


def _dispatch_config_show(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.config_cmd import run_config_show

    run_config_show(
        args.profile,
        args.json,
        config_file=args.config,
        overrides=_runtime_overrides(args),
    )


def _dispatch_inspect(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.model import run_inspect

    run_inspect(args.model, args.device, args.config_json)


def _dispatch_plugins(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.plugins import run_list

    run_list()


def _dispatch_bench_latency(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.bench import run_latency

    run_latency(
        args.world,
        args.steps,
        args.height,
        args.width,
        args.device,
        profile=args.profile,
        overrides=_runtime_overrides(args),
        config_file=args.config,
    )


def _dispatch_bench_throughput(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.bench import run_throughput

    run_throughput(
        args.world,
        args.sessions,
        args.steps,
        args.height,
        args.width,
        args.device,
        profile=args.profile,
        overrides=_runtime_overrides(args),
        config_file=args.config,
    )


def _dispatch_bench_startup(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.bench import run_startup

    run_startup(args.world, args.device)


def _dispatch_bench_vram(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.bench import run_vram

    run_vram(args.world, args.device, args.resolutions)


def _dispatch_bench_profile(args: argparse.Namespace) -> None:
    from worldkernels.cli.commands.bench import run_profile

    run_profile(args.world, args.steps, args.height, args.width, args.device, args.trace)


def _add_bench_common(p: argparse.ArgumentParser, device: str) -> None:
    p.add_argument("--world", default="dummy", help="World model to benchmark.")
    p.add_argument("--device", default=device)


def _add_bench_shape(p: argparse.ArgumentParser, steps: int) -> None:
    p.add_argument("--steps", type=int, default=steps)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)


def build_parser() -> FlexibleArgumentParser:
    parent = _global_parent()
    parser = FlexibleArgumentParser(
        prog="worldkernels",
        description=_DESCRIPTION,
        parents=[parent],
        epilog=WK_SUBCMD_EPILOG,
    )
    parser.set_defaults(dispatch=None)
    parser.add_argument("-V", "--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add_verb(name: str, help_text: str, dispatch: Any) -> FlexibleArgumentParser:
        p = sub.add_parser(
            name,
            help=help_text,
            description=help_text,
            parents=[parent],
            epilog=WK_SUBCMD_EPILOG,
        )
        p.set_defaults(dispatch=dispatch)
        return p

    serve = add_verb("serve", "Start the WorldKernels HTTP/WebSocket server.", _dispatch_serve)
    serve.add_argument("model", nargs="?", default=None, help=f"Model to pre-load. {_MODEL_HELP}")
    frontend = serve.add_argument_group("Frontend")
    frontend.add_argument("--host", default="0.0.0.0", help="Bind address.")
    frontend.add_argument("--port", type=int, default=8000, help="Bind port.")
    frontend.add_argument("--max-sessions", type=int, default=4, help="Concurrent session cap.")
    frontend.add_argument("-k", "--api-key", default=None, help="Require this bearer token.")
    frontend.add_argument("--allowed-origins", default="*", help="Comma-separated CORS origins.")
    frontend.add_argument("--allowed-methods", default="*", help="Comma-separated CORS methods.")
    frontend.add_argument("--allowed-headers", default="*", help="Comma-separated CORS headers.")
    frontend.add_argument(
        "--allow-credentials", action="store_true", help="Allow CORS credentials."
    )
    frontend.add_argument("--ssl-keyfile", default=None, help="TLS key file; enables https.")
    frontend.add_argument("--ssl-certfile", default=None, help="TLS certificate file.")
    frontend.add_argument(
        "--uvicorn-log-level",
        choices=("debug", "info", "warning", "error", "critical", "trace"),
        default="info",
        help="uvicorn log level.",
    )
    frontend.add_argument(
        "--disable-access-log", action="store_true", help="Disable uvicorn access log."
    )
    frontend.add_argument("--root-path", default="", help="ASGI root_path when behind a proxy.")
    serve.add_argument("--device", default="cuda")
    _add_model_args(serve)
    _add_config_args(serve)

    run = add_verb(
        "run",
        "Headless run: bootstrap a model, step N times, optionally save frames/video.",
        _dispatch_run,
    )
    run.add_argument("model", nargs="?", default="dummy", help=_MODEL_HELP)
    run.add_argument("--steps", type=int, default=10, help="Number of world steps.")
    run.add_argument("--action-type", default="null")
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--width", type=int, default=848)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("-o", "--output-dir", default=None, help="Directory for saved outputs.")
    run.add_argument("--output-format", choices=("frames", "video", "both"), default="frames")
    run.add_argument("--fps", type=int, default=24)
    run.add_argument("--video-codec", default="libx264")
    run.add_argument("--modalities", default="frames", help="Comma-separated modalities.")
    run.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--prompt", default=None, help="Initial text prompt.")
    run.add_argument("-i", "--image", default=None, help="Initial image path.")
    run.add_argument(
        "--actions",
        default=None,
        help="Path to a .npy of GR-1 action chunks ([T,384] or [N,12,384]) to drive an "
        "action-conditioned rollout (see examples/gr1_prepare.py). Default: null actions.",
    )
    run.add_argument(
        "--text-encoder",
        choices=("auto", "off", "on"),
        default=None,
        help="'on' loads the reason1-7B encoder (CPU-offloaded) for real prompt conditioning; "
        "default off uses neutral text and relies on image + action conditioning.",
    )
    _add_model_args(run)
    _add_config_args(run)

    pull = add_verb(
        "pull",
        "Pre-fetch a model: install deps, clone packages, download weights. No server.",
        _dispatch_pull,
    )
    pull.add_argument("model", nargs="?", default="dummy", help=_MODEL_HELP)
    pull.add_argument("--variant", default=None, help="Model variant (e.g. 2b_gr1).")
    pull.add_argument("--ckpt-path", default=None, help="Local checkpoint path override.")

    models = add_verb(
        "models",
        "List models: locally cached by default, or the full hub with --all.",
        _dispatch_models,
    )
    models.add_argument("--all", action="store_true", help="Show the full hub catalog.")

    rm = add_verb("rm", "Remove a model from the local cache (weights + manifest).", _dispatch_rm)
    rm.add_argument("model", help=_MODEL_HELP)
    rm.add_argument("--variant", default=None, help="Only remove this variant.")

    add_verb(
        "collect-env",
        "Collect environment info: GPU, deps, plugins, hub, local cache, isolated envs.",
        _dispatch_collect_env,
    )

    config_show = add_verb(
        "config-show",
        "Show the resolved runtime config (component toggles) and each flag's source.",
        _dispatch_config_show,
    )
    config_show.add_argument("--json", action="store_true", help="Emit JSON.")
    _add_config_args(config_show)

    inspect = add_verb(
        "inspect",
        "Show model metadata: transition mode, VRAM estimate, capabilities.",
        _dispatch_inspect,
    )
    inspect.add_argument("model", nargs="?", default="dummy", help=_MODEL_HELP)
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--config-json", default=None, help="WorldConfig overrides as JSON.")

    bench = add_verb(
        "bench", "Benchmarks: latency, throughput, startup, vram, profile.", lambda a: None
    )
    bsub = bench.add_subparsers(dest="bench_command", metavar="<benchmark>", required=True)

    def add_bench(name: str, help_text: str, dispatch: Any) -> FlexibleArgumentParser:
        p = bsub.add_parser(
            name,
            help=help_text,
            description=help_text,
            parents=[parent],
            epilog=WK_SUBCMD_EPILOG,
        )
        p.set_defaults(dispatch=dispatch)
        return p

    latency = add_bench("latency", "Measure single-session step latency.", _dispatch_bench_latency)
    _add_bench_common(latency, "cpu")
    _add_bench_shape(latency, 100)
    _add_config_args(latency)

    throughput = add_bench(
        "throughput", "Measure multi-session concurrent throughput.", _dispatch_bench_throughput
    )
    _add_bench_common(throughput, "cpu")
    _add_bench_shape(throughput, 50)
    throughput.add_argument("--sessions", type=int, default=4)
    _add_config_args(throughput)

    startup = add_bench("startup", "Measure model load + warmup time.", _dispatch_bench_startup)
    _add_bench_common(startup, "cpu")

    vram = add_bench("vram", "Profile VRAM estimates across resolutions.", _dispatch_bench_vram)
    _add_bench_common(vram, "cuda")
    vram.add_argument("--resolutions", default="256x256,480x848,720x1280")

    profile = add_bench(
        "profile", "Run steps under torch.profiler and emit a trace file.", _dispatch_bench_profile
    )
    _add_bench_common(profile, "cuda")
    _add_bench_shape(profile, 10)
    profile.add_argument("--trace", default="wk_profile", help="Chrome trace output basename.")

    add_verb("plugins", "List discovered entry_point plugins.", _dispatch_plugins)

    return parser


def _sub_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _frontend_tokens(frontend: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in frontend.items():
        name = str(key).replace("_", "-").lstrip("-")
        if isinstance(value, bool):
            out.append(f"--{name}" if value else f"--no-{name}")
        elif isinstance(value, (list, dict)):
            raise ValueError(f"config file key {key!r}: nested values are engine-config only")
        else:
            out.extend((f"--{name}", str(value)))
    return out


_NO_FRONTEND_INJECTION = frozenset({"config-show"})


def _expand_config_arg(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    r"""Insert a ``--config`` file's frontend keys as argv tokens.

    Tokens land right after the (sub)command so explicit CLI flags always win;
    engine keys stay in the file and reach ``resolve_runtime_config`` as the
    config-file layer via the ``--config`` flag itself. ``config-show`` only
    resolves the engine layer, so frontend keys are not injected there.
    """
    path: str | None = None
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            break
        if tok.startswith("--config="):
            path = tok.split("=", 1)[1]
            break
    if path is None:
        return argv

    from worldkernels.config.profiles import split_config_file

    frontend, _ = split_config_file(path)
    if not frontend:
        return argv
    insert_at = 0
    choices = _sub_choices(parser)
    for i, tok in enumerate(argv):
        if tok in choices:
            if tok in _NO_FRONTEND_INJECTION:
                return argv
            insert_at = i + 1
            choices = _sub_choices(choices[tok])
            if not choices:
                break
    return argv[:insert_at] + _frontend_tokens(frontend) + argv[insert_at:]


def _configure_ui(args: argparse.Namespace) -> None:
    from worldkernels.ui import configure_logging, set_mode
    from worldkernels.ui.verbosity import OutputMode, Verbosity, set_output_mode, set_verbosity

    for name, default in (("quiet", False), ("verbose", 0), ("debug", False), ("output", "auto")):
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.quiet:
        verbosity = Verbosity.QUIET
    elif args.debug or (args.verbose or 0) >= 2:
        verbosity = Verbosity.DEBUG
    elif args.verbose == 1:
        verbosity = Verbosity.VERBOSE
    else:
        verbosity = Verbosity.NORMAL
    set_verbosity(verbosity)
    set_output_mode(OutputMode(args.output))
    set_mode(_COMMAND_MODE.get(getattr(args, "command", None) or "", "serve"))
    configure_logging(verbosity)


def app(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_expand_config_arg(parser, argv))
    _configure_ui(args)
    if args.dispatch is None:
        from worldkernels import ui

        ui.banner()
        parser.print_help()
        return
    args.dispatch(args)


if __name__ == "__main__":
    app()
