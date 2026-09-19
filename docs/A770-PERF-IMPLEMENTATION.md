# A770 hybrid-model performance implementation plan

Date: 2026-09-18. Status: profiling harness and pinned FLA experiment prepared;
hardware candidate execution remains required before any default changes.

## Objective and evidence

Reduce Qwen3.5-4B scoring latency on the A770 while preserving the scoring
method, pinned model revision, BF16 precision, and validated cache behavior.
Read [A770-PERF-GAP.md](A770-PERF-GAP.md), [INTEL_ARC.md](INTEL_ARC.md), and
[SPEC_REVIEW.md](SPEC_REVIEW.md) before implementation.

The recorded hybrid-model gap is approximately 5–9x versus the RTX 3090,
compared with approximately 1.2x for the dense reranker. The current hybrid
path reports reference convolution and delta-rule implementations, plus a CPU
fallback for `torch.linalg.solve_triangular`. These are strong suspects, but
their individual contributions have not been profiled. Do not promise a
particular speedup or hardware parity from these observations alone.

The dense reranker has unresolved numerical drift. Use its timing only as
context, not as evidence that the XPU backend is generally correct.

## Constraints

- Follow repository `AGENTS.md`. Run from the repository root in an isolated
  environment installed with `pip install -e '.[test]'`.
- Preserve `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, tokenizer, prompt construction,
  option ordering, and conditional option-logit softmax semantics.
- Keep the validated `torch==2.10.0+xpu` / `transformers==5.17.0` environment
  intact. Evaluate dependencies in a separate environment and record exact
  package versions and source commits. Treat any necessary stack upgrade as
  a separate variable, with its own baseline.
- Keep BF16 and the 1024-token PyTorch XPU chunking workaround. Performance
  work must not silently change precision or remove correctness protections.
- Verify the selected device is the A770; device indices can change. Expose
  only that XPU for XPU experiments. For CUDA comparison runs, expose exactly
  one CUDA GPU per scorer process.
- Use new, create-only output paths for all traces, reports, and row-level
  predictions. Do not commit weights, caches, or third-party raw records.
- Do not change headline claims or `results/phase1-summary.json` without the
  supporting row-level evidence, regenerated raw report, updated
  `results/raw/SHA256SUMS`, and updated method/results text.
- Keep unrelated working-tree edits and the static browser demo untouched.

## 1. Establish the baseline and profile

Record the repository revision, environment, driver, GPU identity, model
revision, attention implementation, and active kernel dispatch. Inventory
the existing workspace `runs/` artifacts referenced by the performance note;
do not overwrite them or assume that their environments still match.

Warm the model, then profile representative direct, serial, and shared
decisions. Include short inputs and an approximately 1800-token input through
the existing safe chunked path. Capture CPU and XPU activities and a timeline,
not only an operator table sorted by device time. Attribute wall time to:

- delta-rule scan, including triangular solve and device/host copies;
- reference causal convolution;
- attention and matrix multiplication;
- synchronization, launch overhead, and cache copying or expansion.

Separate model loading and compilation from warm inference. Synchronize
around measured GPU work. Use repeated unprofiled timings to quantify noise
and profiler overhead. Record desktop GPU activity when interpreting variance.
Avoid double-counting nested profiler events or overlapping CPU/GPU activity.

Estimate the maximum useful gain before choosing a patch. If fraction `f` of
wall time is improved by factor `s`, the estimated overall speedup is
`1 / ((1 - f) + f / s)`. Report assumptions and measured fractions.

## 2. Preferred candidate: FLA on PyTorch XPU

Upstream Flash Linear Attention now documents an XPU backend installation
path. This establishes a candidate, not proven compatibility with the A770,
the pinned stack, or every kernel used by this model.

1. Inspect the pinned Transformers Qwen3.5 implementation and its optional
   dependency checks. Identify exactly which imports and dispatch conditions
   enable chunked and recurrent GatedDeltaNet kernels. Check whether missing
   convolution support gates other fast paths.
2. Select and pin a compatible FLA release/commit and Intel Triton build in
   the experimental environment. Inspect dependency resolution before
   installation so CUDA wheels do not replace the XPU stack.
3. Exercise the actual model's prefill and cache-fed recurrence shapes.
   Confirm from traces that the intended XPU kernels run; import success or
   disappearance of warnings is insufficient evidence.
4. Check whether the fused path eliminates the CPU triangular-solve fallback.
   If it does, do not also implement a redundant solve replacement.
5. Validate numerical behavior and cache continuity before integrating a
   narrow, explicit backend choice. Preserve the established fallback path.

If FLA is incompatible, record the exact failure and required stack changes.
Do not spend an unbounded effort porting kernels before evaluating the other
candidate below.

## 3. Alternative candidate: OpenVINO

OpenVINO 2026.2 introduced Qwen3.5 CPU/GPU support. The 2026.4.0 release,
published September 16, adds Qwen3.5 speculative decoding and GPU model
caching for linear-attention models. Executing the model through OpenVINO
could bypass the current PyTorch reference scan and CPU fallback. That is an
inference to test, not a demonstrated A770 speedup.

OpenVINO requires a separate inference backend and model conversion;
installing it does not accelerate the existing PyTorch scorer. Its new
speculative decoding feature concerns generation, while this task primarily
reads next-token option logits. Model caching concerns loading, not warm
scoring throughput. Neither feature alone establishes a remedy here.

### Feasibility gates, in order

1. **Precision:** verify actual GPU execution precision for this model on the
   A770. OpenVINO documents GPU FP16/FP32 inference precision, so BF16 support
   cannot be assumed from the source weights or an inference precision hint.
   Inspect device capabilities and compiled execution information. If BF16
   cannot be preserved, classify this candidate as blocked under the current
   requirement. Describe an FP16/INT4 experiment separately for review; do not
   silently substitute it or present it as a matched BF16 result.
2. **Export:** determine compatible OpenVINO, Optimum Intel, Transformers, and
   tokenizer versions in a separate environment. Export the exact pinned
   source model without quantization or automatic weight compression. Record
   exporter settings and artifact hashes. Do not substitute a preconverted
   model whose provenance or precision differs.
3. **Direct readout:** prove that the backend exposes the required last-token
   vocabulary or exact option logits. Reuse identical token IDs, masks, and
   position semantics. Generated letters and generation log probabilities
   are not substitutes for the scorer's option-logit readout.
4. **State:** establish reset, continuation, snapshot, and independent branch
   behavior for attention KV state, convolution state, and GatedDeltaNet
   recurrent state. Validate direct scoring first, then serial, then shared.
   Generic prefix caching is not proof that shared branch semantics match.
5. **Device and lengths:** select the A770 explicitly and verify execution
   placement. Test long inputs and repeated requests. Do not assume the
   PyTorch long-forward defect applies to OpenVINO, or that OpenVINO's model
   support proves long-input correctness.
6. **Performance:** compare equivalent warm scoring work and numerical outputs
   before considering integration. Record compilation/loading separately.

Keep a feasible OpenVINO adapter opt-in until validation is complete. Specify
unsupported scoring modes explicitly rather than silently recomputing a
different method and labeling it shared scoring.

## 4. Targeted fallback work

If FLA fails and profiling identifies the CPU solve as dominant, investigate
an XPU-resident solve using supported operations or a small dedicated kernel.
First inspect the exact triangular system, chunk sizes, dtypes, and recurrence
dependencies. Compare solve residuals and downstream state/logits against the
reference. Avoid explicit matrix inversion and large Python-loop launch costs.
Existing CPU chunk/cache tests are useful but do not validate a replacement
hybrid-model kernel; add focused numerical and real-XPU coverage.

Optimize convolution only if its remaining measured cost warrants it. Check
prefill and incremental state updates, padding, groups, activation, and dtype.
Consider compiling the isolated GDN block only after fallback operations are
resolved; measure compilation, recompilation, and steady-state latency.
SDPA tuning, B580 comparison, and changing the chunk limit are lower priority.

## 5. Validation and acceptance

Before comparing candidates, record numerical acceptance criteria and the
baseline repeatability. Do not widen tolerances after seeing candidate output.
Compare against the current A770 baseline and published NVIDIA rows, keeping
existing cross-vendor drift distinct from new backend-induced drift.

- Cover lengths 1024, 1025, near the reported 1813-token boundary, and 2049;
  cached suffixes; independent branches; batched branches; and state resets.
- Compare option logits, probabilities, choices, and applicable recurrent
  state. Check finite outputs and normalization, but do not treat those alone
  as correctness. Inspect near-tie choice changes individually.
- Preserve prompt hashes and token counts. Confirm that a faster candidate
  did not truncate inputs or score fewer decisions.
- Use short representative runs first, then all 777 shape decisions in fresh,
  serial, and shared modes for a successful candidate. Report repeat counts,
  median latency, variability, memory, and row-level agreement.
- Require a reproducible gain beyond timing noise with no unexplained
  correctness regression. Report unsupported modes and precision limitations.

Run the required repository checks from the isolated environment:

```bash
pytest -q
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

These checks protect existing behavior and published artifacts; they do not
replace actual A770 numerical and performance validation.

## Agent deliverables

First produce a review of this plan identifying compatibility gaps and the
selected experiment. For subsequent implementation, provide a reproducible
environment manifest, profiling evidence, a narrowly scoped opt-in change,
focused tests, and create-only comparison artifacts. Document unsuccessful
candidates with their exact blockers. Keep correctness fixes, dependency
changes, and performance integration independently reviewable. If no candidate
meets the precision/correctness gates, report that result without changing
the validated default or claiming the gap is solved.

## Implementation status

The selected first experiment is FLA on the pinned PyTorch XPU stack. The
compatibility review found that Transformers 5.17.0 resolves
`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` from the `fla`
package at module import, while both causal-convolution hooks independently
resolve from the CUDA-only `causal_conv1d` package. FLA therefore can replace
the reference delta-rule and its triangular solves without pretending to fix
convolution. Prefill uses the chunked hook; one-token cached recurrence uses
the fused recurrent hook. The validated baseline resolves all four hooks to
their readable PyTorch fallbacks.

`benchmarks/a770_profile.py` implements the baseline and candidate evidence
runner. It preserves the pinned model, BF16, attention selection, prompt and
readout implementations, and 1024-token XPU protection. It rejects non-A770
devices and multiple visible XPUs, warms outside measurement, synchronizes
repeated timings, captures CPU/XPU timelines, records resolved dispatch and a
complete package inventory, and compares profiled option logits,
probabilities, choices, prompt hashes, and token counts with an unprofiled run.
Outputs are isolated in a newly created directory. Inclusive profiler totals
are explicitly not reported as wall fractions.

`benchmarks/manifests/a770-fla-experiment.json` pins the candidate source
commit and existing XPU Triton distribution. The install command uses
`--no-deps` in a separate environment so a resolver cannot replace the pinned
XPU stack with CUDA or another Torch build. This is an experiment pin, not a
compatibility claim. Run the baseline first, then the candidate with
`--require-fla`; retain exact import, compilation, kernel-trace, numerical, and
timing failures if a gate fails. No OpenVINO adapter, solve replacement,
headline result, published artifact, or validated default has been changed.

The first local baseline execution was interrupted during model loading before
profiling began. A retry on the selected A770 failed while materializing the
checkpoint with Level Zero error 20, `UR_RESULT_ERROR_DEVICE_LOST`. A later
attempt died during loading from an earlyoom SIGTERM: the concurrent CPU
reranker run in another session left about 7 GB available with swap nearly
full, below the earlyoom SIGTERM limits. The kernel log from the earlier
window also shows Xe device coredumps and GT0 resets on the A770, consistent
with the reported device loss. No baseline timing or candidate result is
claimed from any of these attempts. The runner writes `failure.json` when
model loading fails, so later create-only runs retain the exact environment
and blocker. These runtime and memory failures do not establish FLA
compatibility or incompatibility.

The FLA import gate has since passed in the experimental environment. A
hard-link clone of the validated venv received FLA 0.6.0 at the pinned commit
plus einops 0.8.2 with `--no-deps`. Transformers 5.17.0 then dispatched both
`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` to FLA while
both causal-convolution hooks stayed on their PyTorch fallbacks, exactly as
predicted. One blocker was root-caused and worked around without patching
binaries: under a Level-Zero-only `ONEAPI_DEVICE_SELECTOR`, importing FLA
aborts inside triton-xpu 3.6.0 because `driver.c` `init_devices` probes for
the OpenCL-backend twin device, which the filter excludes; the C++ exception
cannot be caught from Python. Using
`ONEAPI_DEVICE_SELECTOR="opencl:1;level_zero:1"` satisfies the probe while
PyTorch still sees exactly one device, the A770, which the runner verifies by
name. These are import and dispatch facts only; kernel compilation, numerics,
and speed on the A770 remain unproven. One process error was corrected: a
cloned `pip` script retained the original venv shebang and briefly installed
the candidate packages into the validated environment; both were uninstalled
and the pins re-verified before any further use. Experimental installs must
invoke `<venv>/bin/python -m pip`, never a cloned pip script.

The remaining blocker is host memory, not code: the concurrent CPU reranker
run must finish before a 4B-model load can run safely. When memory allows,
run the baseline with the Level-Zero selector, then the candidate in
`venv-fla` with the dual-backend selector and `--require-fla`, each in its own
create-only output directory.

## Upstream references

Checked September 18, 2026. Recheck compatibility and pin exact versions before
implementation; upstream support does not prove support for this configuration.

- [FLA backend installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
- [OpenVINO release history](https://github.com/openvinotoolkit/openvino/releases)
- [OpenVINO 2026.4.0](https://github.com/openvinotoolkit/openvino/releases/tag/2026.4.0)
- [OpenVINO detailed release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
- [OpenVINO precision controls and limitations](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html)
