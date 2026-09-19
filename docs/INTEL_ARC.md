# Intel Arc XPU Port

This note describes the changes that make SemIf run on an Intel Arc discrete GPU.
The same method can apply to other Intel Arc cards with enough VRAM.

## Hardware

The test system used an Intel Arc A770 with 16 GB of VRAM.
The system also had an Intel Arc B580.
The A770 needed an explicit Level Zero device selector.

Use this command to select the A770 on the test system:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:1
```

Use this command to select only the first visible Intel GPU:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

The correct index can change on another host.
Check the device name before you start a benchmark.

## Python Environment

The validated environment used these package versions:

- `torch==2.10.0+xpu`
- `transformers==5.17.0`

Install PyTorch from the Intel XPU wheel index.
Keep the exact model revisions that the repository specifies.
Do not use `/tmp` for large caches on a system where `/tmp` uses RAM.

## Code Changes

The loader now accepts one visible CUDA GPU or one visible XPU GPU.
It sends the model to the selected device with `device_map`.
It uses `sdpa` attention on both backends.

The scorer now synchronizes CUDA and XPU devices.
This gives correct timing on both backends.

The benchmark scripts now use the active accelerator module.
They no longer call CUDA-only memory and timing functions.

## Qwen3.5-4B XPU Workaround

On XPU, Qwen3.5-4B can corrupt the last-token logits in one long forward.
The failure starts near 1813 input tokens on the A770 test system.
The failure affects the hybrid Qwen3.5 architecture.
Dense Qwen3 models did not show this failure in the tests.

The direct scorer uses a workaround on XPU only.
The serial and shared scorers use the same workaround.
If one forward would exceed 1024 tokens, the scorer splits it into short steps.
Each step uses the model KV cache.
The final logits come from the last step.

This workaround gives the same CPU result as one full forward.
It also avoids the XPU corruption on the A770.
The workaround changes the method used by `direct` on XPU.
The result is still a direct option-logit score.

## Known Limits

The decision, serial, and shared scorers all split long XPU forwards through
the KV cache.
Other code paths that call one long XPU forward can still fail.

The compact generation benchmark needs two settings on XPU:

```bash
--prefill-chunk-size 512 --attention eager
```

Chunked prefill keeps every prefill forward below the corruption length.
Eager attention avoids a separate decode-side collapse that repeats token 0
under SDPA. With both settings, generation reproduced the published NVIDIA
array exactly. Chunk size 1024 produced a 20-item array instead of 21, so the
array terminator sits on a numerics near-tie. Chunk sizes above 1812 are not
safe on XPU.

The direct scorer is the validated A770 path.
The reranker benchmark ran on A770, but its choices drifted from the NVIDIA output.
A CPU fp32 reference over 60 rows showed that the NVIDIA choices match the
CPU reference within drift, and the A770 choices do not.
The reranker readout differences two large yes/no logits, near 17 in
magnitude, and their difference is small, near 0.1 to 1.0.
Small bf16 rounding differences therefore flip many choices.
The drift is model readout sensitivity, not a port correctness bug.
Do not use the A770 reranker result as a matched reproduction.

## Checks

The unit tests passed on the ported code.
The published raw results also passed the repository integrity checks.

The A770 `shape777.py` run used `torch==2.10.0+xpu`.
It compared against the published NVIDIA row-level predictions.

| Mode | Argmax flips | Maximum probability difference |
|---|---:|---:|
| `fresh` | 7 / 777 | 0.0865 |
| `serial_prefix` | 4 / 777 | 0.0622 |
| `parallel_shared` | 3 / 777 | 0.1131 |

A later rerun with the serial and shared prefill paths split gave the same
drift grade. Shared scored 6 / 777 different choices with maximum difference
0.0927. Serial scored 4 / 777 different choices with maximum difference
0.0912.

The published NVIDIA run has 5 / 777 flips between fresh and serial modes.
The A770 result is within the same drift range for this BF16 workload.

The A770 reranker run completed without a crash.
It did not reproduce the NVIDIA row-level choices closely.
Comparing matching pair batch sizes gives 138 / 777 differing choices at size 1,
329 / 777 at size 4, and 341 / 777 at size 8. The cause is unresolved; completion
and finite probabilities alone do not validate this path's numerical behavior.

A later compact-generation run with chunked prefill and eager attention
produced the same complete 21-item array in all three repeats. Its choices
match the published NVIDIA array exactly. Agreement with direct argmax is
18 / 21, the published value. Generation took 14.4 s at median against the
published 5.3 s on the RTX 3090.

See [the specification review](SPEC_REVIEW.md) for the audit scope, fixes, and
the distinction between existing full benchmark evidence and fresh smoke tests.

## Commands

Run the direct benchmark with one visible A770 device:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output /path/to/new-shape777-a770.json
```

Run the reranker benchmark with one visible A770 device:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output /path/to/new-reranker-a770.json
```

Run the repository checks after code changes:

```bash
pytest -q
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```
