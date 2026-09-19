"""Profile the pinned Qwen3.5 scorer paths on one explicitly selected A770."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from semif_phase1.core import load_causal_model, synchronize_device
from semif_phase1.direct import score
from semif_phase1.serial import SerialPrefixScorer
from semif_phase1.shared import score_shared

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
KERNEL_FUNCTIONS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def amdahl_speedup(fraction: float, component_speedup: float) -> float:
    if not 0 <= fraction <= 1 or component_speedup <= 0:
        raise ValueError("Amdahl inputs require 0 <= fraction <= 1 and speedup > 0")
    return 1 / ((1 - fraction) + fraction / component_speedup)


def kernel_dispatch() -> dict[str, dict]:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    result = {}
    for name in KERNEL_FUNCTIONS:
        function = getattr(modeling_qwen3_5, name)
        closure = inspect.getclosurevars(function).nonlocals
        implementation = closure.get("implementation", function)
        fallback = closure.get("torch_function")
        result[name] = {
            "requested_function": closure.get("func_name", name),
            "package": closure.get("package"),
            "implementation_module": implementation.__module__,
            "implementation_name": implementation.__name__,
            "implementation_source": inspect.getsourcefile(implementation),
            "optimized": fallback is not None and implementation is not fallback,
        }
    return result


def _environment() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    try:
        repository_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        repository_revision, dirty = None, None
    return {
        "repository_revision": repository_revision,
        "repository_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "device_selector": os.environ.get("ONEAPI_DEVICE_SELECTOR"),
    }


def _device_manifest(torch, device) -> dict:
    if device.type != "xpu" or torch.xpu.device_count() != 1:
        raise RuntimeError("Expose exactly one XPU device for this A770 experiment")
    properties = torch.xpu.get_device_properties(device)
    if "A770" not in properties.name:
        raise RuntimeError(f"Selected XPU is not an A770: {properties.name}")
    names = (
        "name", "platform_name", "device_id", "uuid", "driver_version", "total_memory",
        "max_compute_units", "gpu_eu_count", "gpu_subslice_count", "has_bfloat16_conversions",
    )
    return {name: str(getattr(properties, name)) for name in names if hasattr(properties, name)}


def _profile_events(profiler) -> list[dict]:
    events = []
    for event in profiler.key_averages():
        device_total = getattr(event, "device_time_total", getattr(event, "xpu_time_total", 0.0))
        self_device = getattr(event, "self_device_time_total", getattr(event, "self_xpu_time_total", 0.0))
        events.append({
            "name": event.key,
            "calls": event.count,
            "cpu_time_total_us": event.cpu_time_total,
            "self_cpu_time_total_us": event.self_cpu_time_total,
            "device_time_total_us": device_total,
            "self_device_time_total_us": self_device,
        })
    return sorted(events, key=lambda item: item["self_device_time_total_us"], reverse=True)


def _operator_evidence(events: list[dict]) -> dict[str, list[str]]:
    patterns = {
        "delta_rule_and_solve": ("gated_delta", "solve_triangular", "wy_fast", "recurrent"),
        "causal_convolution": ("causal_conv", "conv1d", "convolution"),
        "attention_and_matmul": ("scaled_dot_product", "attention", "matmul", "aten::mm", "aten::bmm"),
        "copies_and_synchronization": ("copy", "synchron", "device_to_host", "memcpy"),
    }
    return {
        category: [event["name"] for event in events if any(term in event["name"].lower() for term in terms)]
        for category, terms in patterns.items()
    }


def _comparison(before: list[dict], after: list[dict]) -> dict:
    if [row["id"] for row in before] != [row["id"] for row in after]:
        raise RuntimeError("Profiled output IDs differ from the unprofiled run")
    maximum_logit = maximum_probability = 0.0
    flips = []
    metadata_equal = True
    for left, right in zip(before, after):
        maximum_logit = max(maximum_logit, *(abs(a - b) for a, b in zip(left["option_logits"], right["option_logits"])))
        maximum_probability = max(
            maximum_probability, *(abs(a - b) for a, b in zip(left["probabilities"], right["probabilities"]))
        )
        left_choice = max(range(len(left["probabilities"])), key=left["probabilities"].__getitem__)
        right_choice = max(range(len(right["probabilities"])), key=right["probabilities"].__getitem__)
        if left_choice != right_choice:
            flips.append(left["id"])
        metadata_equal &= all(left[key] == right[key] for key in ("option_ids", "input_tokens", "prompt_sha256"))
    return {
        "max_option_logit_difference": maximum_logit,
        "max_probability_difference": maximum_probability,
        "argmax_flips": flips,
        "prompt_tokens_and_options_equal": metadata_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-id")
    parser.add_argument("--modes", default="direct,serial,shared")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--require-fla", action="store_true")
    parser.add_argument("--amdahl-fraction", type=float)
    parser.add_argument("--amdahl-component-speedup", type=float)
    args = parser.parse_args()
    modes = args.modes.split(",")
    if not modes or any(mode not in {"direct", "serial", "shared"} for mode in modes):
        parser.error("--modes must be a comma-separated subset of direct,serial,shared")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if (args.amdahl_fraction is None) != (args.amdahl_component_speedup is None):
        parser.error("Amdahl fraction and component speedup must be supplied together")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir must be new")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    selected = next((row for row in rows if row["id"] == args.row_id), None) if args.row_id else rows[0]
    if selected is None:
        parser.error(f"Unknown row ID: {args.row_id}")
    group_id = selected.get("group_id")
    group = [row for row in rows if row.get("group_id") == group_id] if group_id is not None else [selected]

    try:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.attention)
    except Exception as error:
        failure = {
            "version": "a770-qwen35-profile-v1",
            "status": "failed",
            "phase": "model_load",
            "error_type": type(error).__name__,
            "error": str(error),
            "model": {"source": args.model, "revision": args.revision, "attention": args.attention},
            "environment": _environment(),
        }
        (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        raise
    import torch

    device = next(model.parameters()).device
    device_manifest = _device_manifest(torch, device)
    dispatch = kernel_dispatch()
    if args.require_fla and not all(dispatch[name]["optimized"] for name in KERNEL_FUNCTIONS[:2]):
        raise RuntimeError("--require-fla requested, but Qwen3.5 did not dispatch both delta-rule paths to FLA")

    serial = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)

    def run(mode: str) -> list[dict]:
        if mode == "direct":
            return [score(model, tokenizer, selected, metadata, args.max_tokens)]
        if mode == "serial":
            return [serial.score(selected)]
        return score_shared(model, tokenizer, group, metadata, args.max_tokens)[0]

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
    results = []
    for mode in modes:
        run(mode)
        synchronize_device(device)
        timings, baseline = [], None
        for _ in range(args.repeats):
            synchronize_device(device)
            started = time.perf_counter()
            value = run(mode)
            synchronize_device(device)
            timings.append(time.perf_counter() - started)
            baseline = baseline or value
        trace_path = args.output_dir / f"{mode}.trace.json"
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            profiled = run(mode)
            synchronize_device(device)
        profiler.export_chrome_trace(str(trace_path))
        events = _profile_events(profiler)
        results.append({
            "mode": mode,
            "repeats": args.repeats,
            "wall_seconds": timings,
            "median_wall_seconds": statistics.median(timings),
            "min_wall_seconds": min(timings),
            "max_wall_seconds": max(timings),
            "profiled_comparison": _comparison(baseline, profiled),
            "profiled_outputs": profiled,
            "trace": trace_path.name,
            "top_events": events[:100],
            "operator_evidence": _operator_evidence(events),
            "timing_note": "Profiler operator totals are inclusive and may overlap; they are not wall-time fractions.",
        })

    report = {
        "version": "a770-qwen35-profile-v1",
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "selected_row_id": selected["id"],
        "selected_group_id": selected.get("group_id"),
        "model": metadata,
        "device": device_manifest,
        "environment": _environment(),
        "kernel_dispatch": dispatch,
        "fla_required": args.require_fla,
        "results": results,
    }
    if args.amdahl_fraction is not None:
        report["amdahl_estimate"] = {
            "measured_wall_fraction": args.amdahl_fraction,
            "assumed_component_speedup": args.amdahl_component_speedup,
            "estimated_overall_speedup": amdahl_speedup(args.amdahl_fraction, args.amdahl_component_speedup),
            "note": "The fraction must come from non-overlapping trace analysis, not summed profiler events.",
        }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output_dir), "timings": [
        {"mode": result["mode"], "median_wall_seconds": result["median_wall_seconds"]} for result in results
    ]}))


if __name__ == "__main__":
    main()
