# A770 specification review — 2026-09-18

The port implements the accelerator plumbing specified in the local workspace
file `A770-PORT-SPEC.md`, which is outside this repository. It is not a fully
validated replacement for every CUDA execution path. Direct scoring has the
strongest A770 evidence. Generation remains unsuccessful, reranker drift
remains unexplained, and long serial/shared forwards bypass the direct
scorer's corruption workaround.

## Remaining findings

1. **High: long serial/shared inputs remain exposed to XPU corruption.**
   `src/semif_phase1/serial.py` calls `_cached_forward` on the whole prefix and
   suffix. `src/semif_phase1/shared.py` likewise calls the model without chunking.
   Both accept full prompts up to 4096 tokens. A state or criterion that pushes
   either call beyond the observed Qwen3.5 failure threshold can therefore
   produce unreliable scores. The successful 777-row fixture is not coverage
   for the entire accepted input range. This is a code-path finding grounded in
   the earlier corruption investigation. This review did not repeat that
   investigation or establish a universal safe threshold.
2. **High: compact generation is not validated on A770.**
   `benchmarks/decision_vs_generation.py` still uses an unsplit `model.generate`
   prefill. The existing A770 report records three invalid arrays, each reaching
   128 output tokens, and null agreement. Its timing ratio must not be treated
   as the published successful generation comparison.
3. **High: reranker agreement is unresolved.** Matching the A770 and NVIDIA
   row-level files by decision ID and pair batch size gives the differences
   below. The original spec explicitly treats drift as informational, so this
   does not fail its benchmark-execution requirement. It does prevent a claim
   that the port is numerically equivalent or validated for decision quality.
   The longest reranker prompt in this fixture is 1755 tokens. The known XPU
   corruption affected single forwards beyond 1812 tokens on the hybrid
   Qwen3.5 model. The reranker uses batched forwards on a dense model below
   that length. This drift therefore needs a separate cause.

## Existing full-run evidence independently checked

These are existing files in the local workspace `runs/` directory, which is
outside this repository. They are not full benchmarks rerun during this review.
Each compared mode has 777 decisions. Parallel A770 `parallel_shared`
maps to published NVIDIA `parallel_suffix`. All comparisons use matching modes,
IDs, and option order.

| Mode | Different choices vs NVIDIA | Maximum probability difference |
|---|---:|---:|
| Direct fresh | 7 / 777 | 0.086512 |
| Serial prefix | 4 / 777 | 0.062177 |
| Parallel shared | 3 / 777 | 0.113118 |
| Reranker, pair batch 1 | 138 / 777 | 0.839696 |
| Reranker, pair batch 4 | 329 / 777 | 0.767278 |
| Reranker, pair batch 8 | 341 / 777 | 0.777158 |

Sources are `shape777-a770-t210.json` with its predictions,
`shape777-reranker-a770.json` with its predictions, and
`decision-vs-generation-a770.json`. NVIDIA references are the corresponding
committed `results/raw/` files. No quality threshold or headline claim changed.

## Requirements confirmed

- The current package name is `semif_phase1`, and the CLI is `semif-score`.
  The specification and old resume notes predate the rename.
- The isolated environment has torch `2.10.0+xpu` and transformers `5.17.0`.
  The dependency declarations and frozen model revisions remain unchanged.
- Loader selection supports CUDA and XPU, requires one device in the selected
  backend, uses an explicit device map, and requests SDPA.
- All four scorer paths synchronize the active accelerator. The three named
  benchmarks use its name and memory APIs, preserving `peak_cuda_bytes`.
- A fresh check with `ONEAPI_DEVICE_SELECTOR=level_zero:1` exposes exactly one
  A770, device ID `0x56A0`, UUID `8680a056-0800-0000-0600-000000000000`.
- A fresh allocation of 8 GiB + 4 KiB on that device, with writes and reads at
  both ends and synchronization, succeeds. The historical 4 GiB allocation cap
  does not block that operation.
- Direct XPU scoring splits long input into 1024-token cache steps. Added tests
  compare this branch against a tiny native Qwen3 model's full CPU forward at
  1025 and 2049 tokens, within numerical tolerance. They test cache continuity,
  not real XPU kernel correctness.

## Corrections made in this review

- Both shape runners previously checked only the report filename, then used
  `write_text` for the report and predictions. A new report path could silently
  replace existing predictions. Both now preflight both paths and exclusively
  create every output. The generation report also uses exclusive creation to
  prevent overwrite if another process creates its path after preflight.
  A failed concurrent writer can leave an empty newly created file. The helper
  does not promise an atomic multi-file transaction.
- Generation validation previously raised `TypeError` for JSON arrays containing
  objects or nested arrays. It now records those outputs as invalid.
- Long direct XPU scoring now rejects a model that returns no KV cache, rather
  than silently scoring later chunks without their preceding context.
- Added CPU-only regression coverage for accelerator selection, device count,
  synchronization, cached chunking, artifact collisions, and malformed output.

## Validation scope

The unit and browser-source suite passes: **31 tests**. The browser checks are
static source checks, not a fresh browser inference session. All **15 raw
artifact checksums** and **69 published summary claims** passed the repository's
required integrity checks. Published artifacts were not edited.

Fresh A770 scoring passed on all three owned examples for all four pinned
models, using the pinned revisions and offline cached weights. All returned
probabilities were finite and normalized. Saved outputs live in the local
workspace `runs/` directory, which is outside this repository:
`review-20260918-qwen06-direct.jsonl`, `review-20260918-minicpm5-direct.jsonl`,
`review-20260918-qwen35-4b-direct.jsonl`, and
`review-20260918-reranker-smoke.jsonl`.

An earlier Qwen3.5-4B attempt failed before the successful retry. earlyoom sent
SIGTERM during model loading (exit 143). The system journal confirms available
memory fell to 632 MiB and free swap to zero at 14:44:11 local time. A retry
after available memory recovered completed all three rows. No unrelated process
was stopped. That failure was an environment event, not a model-code failure.
The old full benchmarks were not rerun in this review.

The old `RESUME-STATE.md` describes an intermediate snapshot: its pending full
benchmarks now exist, its package paths are stale, and its 14-test count is
outdated. The full runs above supersede its 21-row provisional direct result.
