# SemIf desk lab

This is a server-backed decision page for one Intel Arc GPU. It mirrors the
browser lab, but the model runs on the local machine through the SemIf
scorers. No WebGPU, no quantized GGUF, no external requests.

The page shows the two execution paths that the repository benchmarks:

1. **Shared prefill.** The server prefills the exact state once. Every
   criterion branches from that cache and reads its option logits in one
   batch. This is `score_shared`.
2. **Fresh scoring.** Every criterion runs its own full forward over the
   whole prompt. This is `direct.score`.

The page displays only timings collected in the current request. Criteria
share one state text, and each criterion holds two to sixteen options.

## Run on the A770

Build the environment once, as the repository README describes. Then start
the server with exactly one visible GPU:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 python arc-demo/server.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Verify the device index before you start. The index maps to a different card
on another host. Open `http://127.0.0.1:8765` in a browser.

Useful options:

- `--attention eager` avoids the SDPA decode collapse on XPU. The scoring
  paths do not need it, but generation experiments on this server would.
- `--host 0.0.0.0` serves the page on your LAN. The default binds
  `127.0.0.1` only. Do not expose a model server to people you do not trust.
- `--port` changes the port. The default is `8765`.

## API

- `GET /api/health` reports the loaded model, revision, dtype, attention
  implementation, and PyTorch version.
- `POST /api/score` accepts one JSON object:

```json
{
  "state": "one exact state text",
  "criteria": [
    {"question": "Which queue?", "options": ["Access", "Billing"]},
    {"question": "Escalate now?", "options": ["Yes", "No"]}
  ],
  "modes": ["shared", "fresh"]
}
```

It returns per-criterion option probabilities and the timing breakdown for
each requested mode. The `fresh` mode costs one full forward per criterion.

## Limits

- The server scores one request at a time. A lock holds the model during
  each request.
- The body limit is one megabyte.
- Probabilities are conditional on the supplied options. They are not
  calibrated confidence. BF16 arithmetic differs between GPU vendors and
  execution paths, so near-tie choices can flip. See `docs/SPEC_REVIEW.md`.
- The server holds no state between requests and writes no files.

## Files

- `server.py` — the server. Standard library HTTP only.
- `index.html`, `style.css`, `app.js` — the page. No build step and no CDN.
  The stylesheet comes from `webgpu-demo/style.css` with a desk-lab addendum.
