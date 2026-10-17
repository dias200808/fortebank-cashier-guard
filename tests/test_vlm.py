import json

import httpx
import numpy as np
import pytest

from app.pipeline.vlm import FakeVLM, OpenAIVLM, VLMError, p_yes

IMG = np.zeros((20, 10, 3), np.uint8)


def response(top):
    return {"choices": [{"message": {"content": top[0]["token"]},
                         "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}]}


def test_p_yes_normalizes_yes_and_no():
    assert abs(p_yes([{"token": "Yes", "logprob": -0.1}, {"token": " no", "logprob": -2.4}]) - 0.909) < 0.01
    assert p_yes([{"token": "Maybe", "logprob": -0.1}]) == 0.5


def test_yes_prob_sends_logprob_request():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=response([{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.4}]))

    vlm = OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    p = vlm.yes_prob([IMG], "Is it?")
    assert 0.89 < p < 0.92
    assert seen["max_tokens"] == 1 and seen["logprobs"] is True and seen["temperature"] == 0.0
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Yes or No" in content[-1]["text"]


def test_retries_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    vlm = OpenAIVLM("http://x/v1", "m", retries=2, backoff=0.0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VLMError):
        vlm.yes_prob([IMG], "Q")
    assert calls["n"] == 3


def test_yes_probs_keeps_order():
    def handler(request):
        q = json.loads(request.content)["messages"][0]["content"][-1]["text"]
        tok = "Yes" if q.startswith("A") else "No"
        return httpx.Response(200, json=response([{"token": tok, "logprob": -0.01}]))

    vlm = OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert vlm.yes_probs([([IMG], "A?"), ([IMG], "B?"), ([IMG], "A2?")]) == [1.0, 0.0, 1.0]


def test_fake_vlm():
    f = FakeVLM(answer=lambda imgs, q: 0.8 if "safe" in q else 0.1, reply=lambda imgs, p: '{"bbox_2d": null}')
    assert f.yes_probs([([IMG], "safe open?"), ([IMG], "door?")]) == [0.8, 0.1]
    assert f.text([IMG], "x") == '{"bbox_2d": null}' and f.calls == 3


def test_hung_server_aborts_batch_quickly():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("hung", request=request)

    vlm = OpenAIVLM("http://x/v1", "m", workers=4, retries=2, backoff=0.0,
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VLMError):
        vlm.yes_probs([([IMG], f"Q{i}?") for i in range(40)])
    assert calls["n"] < 3 * 40


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad model"})

    vlm = OpenAIVLM("http://x/v1", "m", retries=2, backoff=0.0,
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VLMError):
        vlm.yes_prob([IMG], "Q")
    assert calls["n"] == 1


def test_malformed_response_raises_vlm_error():
    vlm = OpenAIVLM("http://x/v1", "m",
                    client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))))
    with pytest.raises(VLMError):
        vlm.yes_prob([IMG], "Q")
    vlm2 = OpenAIVLM("http://x/v1", "m",
                     client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": 1}))))
    with pytest.raises(VLMError):
        vlm2.yes_prob([IMG], "Q")


def test_missing_logprobs_falls_back_and_warns_once(monkeypatch, caplog):
    import logging

    import app.pipeline.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "_warned_no_logprobs", False, raising=False)
    body = {"choices": [{"message": {"content": "Yes"}, "logprobs": None}]}
    vlm = OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))))
    with caplog.at_level(logging.WARNING, logger="app.pipeline.vlm"):
        assert vlm.yes_prob([IMG], "Q") == 1.0 and vlm.yes_prob([IMG], "Q") == 1.0
    assert len([r for r in caplog.records if "logprobs" in r.getMessage()]) == 1


def test_qwen35_turns_thinking_off_and_qwen3vl_sends_nothing_extra():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yes"}, "logprobs": {"content": [
            {"token": "Yes", "logprob": -0.1, "top_logprobs": [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.4}]}]}}]})

    for model in ("Qwen/Qwen3.5-9B", "RedHatAI/Qwen3.5-9B-FP8-dynamic", "Qwen/Qwen3-VL-8B-Instruct-FP8"):
        vlm = OpenAIVLM("http://x/v1", model, workers=1, client=httpx.Client(transport=httpx.MockTransport(handler)))
        assert vlm.yes_probs([([np.zeros((8, 8, 3), np.uint8)], "q?")])[0] > 0.8
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in seen[2]
