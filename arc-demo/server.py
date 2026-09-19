"""Serve the SemIf desk demo from one local accelerator.

Run with exactly one visible GPU, for example:

    ONEAPI_DEVICE_SELECTOR=level_zero:1 python arc-demo/server.py \
        --model Qwen/Qwen3.5-4B --revision 851bf6e...

The server keeps one loaded model in memory. It exposes a small JSON API and
serves the static page from this directory. All scoring stays on the machine.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from semif_phase1.core import load_causal_model, validate_row
from semif_phase1.direct import score as score_fresh
from semif_phase1.shared import score_shared

HERE = Path(__file__).resolve().parent
STATIC_FILES = {"index.html": "text/html; charset=utf-8", "style.css": "text/css; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}
MAX_BODY_BYTES = 1_000_000
MAX_CRITERIA = 16

_lock = threading.Lock()
_state = {}


def build_rows(payload: dict) -> list[dict]:
    """Turn one API payload into validated shared-state rows."""
    state = payload.get("state")
    criteria = payload.get("criteria")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be a nonempty string")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("criteria must be a nonempty list")
    if len(criteria) > MAX_CRITERIA:
        raise ValueError(f"criteria must contain at most {MAX_CRITERIA} entries")
    rows = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError("every criterion must be an object")
        descriptions = criterion.get("options")
        if not isinstance(descriptions, list):
            raise ValueError("every criterion needs an options list")
        row = {
            "id": f"criterion-{index}",
            "state": state,
            "question": criterion.get("question"),
            "options": [{"id": f"o{number}", "description": text} for number, text in enumerate(descriptions)],
        }
        validate_row(row)
        rows.append(row)
    return rows


def _slim(result: dict, criterion: dict) -> dict:
    return {
        "id": result["id"],
        "question": criterion["question"],
        "option_ids": result["option_ids"],
        "option_descriptions": [option if isinstance(option, str) else option["description"] for option in criterion["options"]],
        "probabilities": result["probabilities"],
        "option_logits": result["option_logits"],
        "input_tokens": result["input_tokens"],
    }


def run_score(payload: dict) -> dict:
    """Score every criterion once through the shared prefill, then optionally fresh."""
    rows = build_rows(payload)
    modes = payload.get("modes") or ["shared"]
    if not isinstance(modes, list) or any(mode not in ("shared", "fresh") for mode in modes):
        raise ValueError("modes may contain only shared and fresh")
    if not modes:
        raise ValueError("modes may not be empty")
    reply = {"modes": modes, "criteria": len(rows)}
    with _lock:
        if "shared" in modes:
            results, timing = score_shared(_state["model"], _state["tokenizer"], rows, _state["metadata"])
            reply["shared"] = {
                "results": [_slim(result, criterion) for result, criterion in zip(results, payload["criteria"])],
                "timing": timing,
            }
        if "fresh" in modes:
            started = time.perf_counter()
            scored = []
            for row, criterion in zip(rows, payload["criteria"]):
                result = score_fresh(_state["model"], _state["tokenizer"], row, _state["metadata"])
                scored.append(_slim(result, criterion))
            reply["fresh"] = {
                "results": scored,
                "timing": {"total_seconds": time.perf_counter() - started},
            }
    return reply


def health() -> dict:
    metadata = _state.get("metadata", {})
    return {
        "ready": bool(_state.get("model")),
        "model": metadata.get("source"),
        "revision": metadata.get("revision"),
        "dtype": metadata.get("dtype"),
        "attention": metadata.get("attention"),
        "torch": metadata.get("torch_version"),
        "transformers": metadata.get("transformers_version"),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "semif-desk/1.0"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, allow_nan=False).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json(200, health())
            return
        name = path.lstrip("/") or "index.html"
        if name in STATIC_FILES:
            self._send(200, (HERE / name).read_bytes(), STATIC_FILES[name])
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api/score":
            self._json(404, {"error": "not found"})
            return
        if not _state.get("model"):
            self._json(503, {"error": "model is still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(400, {"error": f"body must hold between 1 and {MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "body must be one JSON object"})
            return
        try:
            self._json(200, run_score(payload))
        except (ValueError, RuntimeError) as error:
            self._json(400, {"error": str(error)})

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        return  # keep the console quiet; errors still surface through responses


def create_server(model, tokenizer, metadata: dict, host: str, port: int) -> ThreadingHTTPServer:
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["metadata"] = metadata
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"loading {args.model}@{args.revision} (attention={args.attention}) ...", flush=True)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.attention)
    httpd = create_server(model, tokenizer, metadata, args.host, args.port)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
