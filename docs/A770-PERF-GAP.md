# A770 performance gap on the Qwen3.5 hybrid path

Date: 2026-09-18. Scope: why the SemIf A770 port is slow on hybrid models, and
what to research. Correctness of the port is not in question here. The
correctness bug (long single forwards) is a separate issue. See
`SPEC_REVIEW.md` and `INTEL_ARC.md` in this directory for that topic.

## 1. Observation

Dense models run close to NVIDIA speed on the A770. The hybrid Qwen3.5 model
runs 5 to 9 times slower than the RTX 3090. The gap is architecture-specific.
It does not come from the port logic, the device selector, or the chunked
forward workaround.

## 2. Measured comparison

NVIDIA numbers are the published RTX 3090 results in `../results/raw/`. A770
numbers come from the workspace `runs/` directory, which is outside this
repository. Same fixture (shape777, 777 decisions), same pinned model
revisions, same prompt builder.

| Path | RTX 3090 | A770 | A770 / 3090 |
|---|---:|---:|---:|
| shape777 fresh direct | 333 s (2.33 dec/s) | 2672 s (0.29 dec/s) | 8.0x |
| shape777 serial | 72 s (10.75 dec/s) | 398 s (1.95 dec/s) | 5.5x |
| shape777 shared | 39 s (20.03 dec/s) | 333 s (2.33 dec/s) | 8.6x |
| Reranker, pair batch 1 | 417 s (1.86 j/s) | 534 s (1.45 j/s) | 1.28x |
| Reranker, pair batch 8 | 435 s (1.79 j/s) | 504 s (1.54 j/s) | 1.16x |
| 21-decision shared readout | 1.02 s | 9.33 s | 9.1x |
| Compact generation | 5.33 s | 16.2 s | n/a (output invalid on A770) |

The reranker uses Qwen3-Reranker-4B, a dense transformer. Every slow row uses
Qwen3.5-4B, the hybrid GatedDeltaNet (GDN) architecture. The dense path pays
about 1.2x. The hybrid path pays 5x to 9x.

## 3. Verified causes

Every A770 model load prints these warnings from transformers:

```text
`causal_conv1d_fn` is falling back to its reference PyTorch implementation
because `causal_conv1d` is not installed.
`chunk_gated_delta_rule` is falling back to its reference PyTorch
implementation because `flash-linear-attention` is not installed.
```

Every forward also prints this warning:

```text
UserWarning: Aten Op fallback from XPU to CPU happends.
  new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
```

Cause 1: `causal_conv1d` is a CUDA-only extension package. There is no XPU
build on this system. The conv step of each GDN layer runs a slow reference
implementation.

Cause 2: `flash-linear-attention` (fla) is not installed and does not import
on XPU here. The core GDN delta-rule scan runs a reference implementation
instead of a fused Triton kernel.

Cause 3: `torch.linalg.solve_triangular` has no XPU operator in torch
2.10.0+xpu. Each call falls back to the CPU. The GDN scan calls it inside its
innermost loop, once per layer and per sequence chunk. Each fallback is a
device-to-host round trip with synchronization.

Cause 4: the correctness workaround splits any forward longer than 1024 tokens
into cache-fed steps. This doubles or triples the forward count for long
prompts in fresh mode. Its measured cost is a factor of about 1.4 on the
serial and shared modes. This is small next to the kernel gap.

Cause 5 (background): the A770 is the boot GPU and shares time with desktop
processes. This adds noise. It does not explain a factor of 8.

## 4. Why the two architectures differ

A dense transformer layer needs attention, matmuls, and pointwise ops. The XPU
backend covers these ops, so the reranker runs near hardware speed.

The GDN layer adds a linear-attention recurrence with a chunked scan, a causal
convolution, and a batched triangular solve. All three of those steps lose
their fast implementations on this stack. The loss multiplies across 36
layers.

## 5. Research questions, ranked by expected payoff

1. **Profile one forward first.** This ranks the three causes by true cost.
   Lowest effort, highest information. See section 6.
2. **Does fla run on Intel GPUs?** fla is Triton-based. Triton has an Intel XPU
   backend in recent versions. Check the fla repository for XPU support,
   open issues, or forks. If a working build exists, install it and re-measure.
   This replaces the slowest step. Expected payoff: the largest single gain.
3. **Can the triangular solve stay on the XPU?** Check the operator coverage
   tracker for the torch XPU backend (the `intel/torch-xpu-ops` repository on
   GitHub lists supported ops per release). Search for
   `linalg_solve_triangular`. If it is absent, file or track a request. A local
   workaround is also possible: express the solve with operators that do exist
   on XPU. Check numerical equivalence before any change. The CPU fallback
   check in this port verifies exact equality on CPU, so any replacement has a
   test harness ready.
4. **Is there an XPU port of `causal_conv1d`?** The upstream package targets
   CUDA. Check for forks and for IPEX coverage. The kernel is small, but it
   runs once per layer.
5. **Does `torch.compile` help on XPU?** Inductor with the Triton XPU backend
   can fuse parts of the reference scan. This is experimental. Compile the GDN
   block alone first. Watch for cache-related failures with dynamic shapes.
6. **Which SDPA backend runs on XPU?** Check `torch.backends.xpu` flags and
   the selected attention implementation. The dense result suggests attention
   itself is not the bottleneck, so this is a low-priority check.
7. **Does the B580 show the same gap?** The B580 is a newer Xe2 GPU. If the
   gap is the same, the cause is the software stack. If the gap is smaller,
   kernel coverage or clocks differ by product. One short rerun answers this.

## 6. Profiling plan

Run one warm forward of Qwen3.5-4B at about 1800 tokens under
`torch.profiler` with `ONEAPI_DEVICE_SELECTOR=level_zero:1`. Record the share
of wall time in:

- `solve_triangular` and its CPU fallback path
- the reference `causal_conv1d_fn`
- the reference `chunk_gated_delta_rule`
- SDPA and matmuls
- host synchronization gaps

Sketch:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    scorer.score(row)          # one warm shared or direct decision
print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))
```

Convert the shares into a predicted post-fix speedup before writing any code.
A fix that addresses 5 percent of the wall time is not worth the risk.

## 7. Non-goals

- Do not change precision. The bf16 pin and the published evidence are fixed.
- Do not tune the chunked-forward workaround. Its cost is already small.
- Do not chase the cross-vendor BF16 drift. It is at NVIDIA's own drift level.
- Do not bundle performance changes with correctness changes. Keep the commit
  history separable, as this port has done so far.

## 8. Artifacts and pointers

- Workspace measurement files (outside this repository):
  `runs/shape777-a770-t210.json`, `runs/shape777-reranker-a770.json`,
  `runs/decision-vs-generation-a770.json`, `runs/sharedfix-regression.log`
- Published NVIDIA references: `../results/raw/shape777-direct.json`,
  `../results/raw/shape777-reranker.json`,
  `../results/raw/decision-vs-compact-array.json`
- Port documentation in this directory: `INTEL_ARC.md`, `SPEC_REVIEW.md`
- Warnings to grep for: `falling back to its reference`, `Aten Op fallback`
- Upstream packages to investigate: `flash-linear-attention` (fla),
  `causal-conv1d` (Dao-AILab), `intel-extension-for-pytorch` (IPEX),
  `intel/torch-xpu-ops` (operator coverage tracker)
