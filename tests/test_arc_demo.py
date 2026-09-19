"""Tests for the arc-demo server and page."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "arc-demo"


def server_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("arc_demo_server", DEMO / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_page_contract():
    html = (DEMO / "index.html").read_text()
    assert 'href="style.css"' in html
    assert '<script src="app.js">' in html
    assert "Intel Arc" in html
    app = (DEMO / "app.js").read_text()
    assert "/api/health" in app and "/api/score" in app
    assert "filled.length !== criterion.options.length" in app
    assert "filled !== criterion.options.length" not in app
    readme = (DEMO / "README.md").read_text()
    assert "ONEAPI_DEVICE_SELECTOR" in readme and "127.0.0.1" in readme
    style = (DEMO / "style.css").read_text()
    assert ".criterion" in style


def test_build_rows_validates_payload():
    module = server_module()
    payload = {
        "state": "one state",
        "criteria": [{"question": "q", "options": ["a", "b"]}],
    }
    rows = module.build_rows(payload)
    assert rows[0]["id"] == "criterion-0"
    assert rows[0]["options"][0] == {"id": "o0", "description": "a"}
    for bad, change in (
        ("empty state", {"state": ""}),
        ("no criteria", {"criteria": []}),
        ("too many criteria", {"criteria": [{"question": "q", "options": ["a", "b"]}] * 17}),
        ("one option", {"criteria": [{"question": "q", "options": ["a"]}]}),
        ("missing question", {"criteria": [{"options": ["a", "b"]}]}),
    ):
        with pytest.raises(ValueError):
            module.build_rows({**payload, **change})


def test_run_score_rejects_unknown_modes():
    module = server_module()
    payload = {"state": "s", "criteria": [{"question": "q", "options": ["a", "b"]}], "modes": ["bogus"]}
    with pytest.raises(ValueError):
        module.run_score(payload)


class ByteTokenizer:
    """Deterministic offline tokenizer: one token per byte."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=False, **kwargs):
        return "<s>" + "".join(f"<{turn['role']}>{turn['content']}</{turn['role']}>" for turn in turns) + "<a>"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")


@pytest.fixture(scope="module")
def tiny_server():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=300, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).eval()
    module = server_module()
    metadata = {
        "source": "tiny/test", "revision": "a" * 40, "dtype": "float32",
        "attention": "sdpa", "torch_version": "test", "transformers_version": "test",
    }
    httpd = module.create_server(model, ByteTokenizer(), metadata, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as reply:
        return reply.status, reply.read()


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_health_and_page(tiny_server):
    status, body = get(f"{tiny_server}/api/health")
    assert status == 200
    health = json.loads(body)
    assert health["ready"] is True and health["model"] == "tiny/test"
    status, page = get(tiny_server + "/")
    assert status == 200 and b"SemIf desk lab" in page


def test_score_round_trip_shared_and_fresh(tiny_server):
    payload = {
        "state": "The lens was ordered but never fitted.",
        "criteria": [
            {"question": "Was the lens fitted?", "options": ["Yes", "No"]},
            {"question": "Is the order open?", "options": ["Yes", "No"]},
        ],
        "modes": ["shared", "fresh"],
    }
    status, reply = post(f"{tiny_server}/api/score", payload)
    assert status == 200, reply
    assert reply["criteria"] == 2
    for mode in ("shared", "fresh"):
        results = reply[mode]["results"]
        assert len(results) == 2
        for result in results:
            assert len(result["probabilities"]) == 2
            assert abs(sum(result["probabilities"]) - 1.0) < 1e-5
    assert reply["shared"]["timing"]["prefix_tokens"] > 0
    assert reply["fresh"]["timing"]["total_seconds"] > 0


def test_score_rejects_bad_payload(tiny_server):
    status, reply = post(f"{tiny_server}/api/score", {"state": "", "criteria": []})
    assert status == 400 and "state" in reply["error"]
    status, reply = post(f"{tiny_server}/api/score", {"state": "s", "criteria": []})
    assert status == 400
