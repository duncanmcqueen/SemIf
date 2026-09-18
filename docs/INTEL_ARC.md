# Intel Arc XPU Port

This note describes the changes that make OpenJEV run on an Intel Arc discrete GPU.
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
If the input is longer than 1024 tokens, it splits the input into short steps.
Each step uses the model KV cache.
The final logits come from the last step.

This workaround gives the same CPU result as one full forward.
It also avoids the XPU corruption on the A770.
The workaround changes the method used by `direct` on XPU.
The result is still a direct option-logit score.

## Known Limits

The workaround applies only to the direct scorer.
Other code paths that call one long XPU forward can still fail.

The compact generation benchmark still has this limit.
The `generate()` call performs its own long prefill.
The current port does not split that internal prefill.

The direct scorer is the validated A770 path.
The reranker benchmark ran on A770, but its choices drifted from the NVIDIA output.
Do not use the A770 reranker result as a matched reproduction.
Longer reranker prompts also need a separate check.

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

The published NVIDIA run has 5 / 777 flips between fresh and serial modes.
The A770 result is within the same drift range for this BF16 workload.

The A770 reranker run completed without a crash.
It did not reproduce the NVIDIA row-level choices closely.

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
