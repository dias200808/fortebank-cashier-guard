"""VLM access. OpenAIVLM talks to a local vLLM OpenAI-compatible server; FakeVLM is for tests."""
import base64
import io
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

import httpx
import numpy as np
from PIL import Image

YES_NO_SUFFIX = "\nAnswer with exactly one word: Yes or No."
log = logging.getLogger(__name__)
_warned_no_logprobs = False


class VLMError(RuntimeError):
    pass


class VLM(Protocol):
    def yes_prob(self, images: list[np.ndarray], question: str) -> float: ...
    def yes_probs(self, items: list[tuple[list[np.ndarray], str]]) -> list[float]: ...
    def text(self, images: list[np.ndarray], prompt: str, max_tokens: int = 256) -> str: ...


def to_data_url(img: np.ndarray, max_side: int = 896) -> str:
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def p_yes(top_logprobs: list[dict]) -> float:
    py = pn = 0.0
    for item in top_logprobs:
        tok = item["token"].strip().strip(".").lower()
        if tok == "yes":
            py += math.exp(item["logprob"])
        elif tok == "no":
            pn += math.exp(item["logprob"])
    return 0.5 if py + pn == 0 else py / (py + pn)


class OpenAIVLM:
    def __init__(self, base_url: str, model: str, workers: int = 16, timeout: float = 30.0,
                 retries: int = 2, backoff: float = 0.5, client: httpx.Client | None = None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.workers, self.retries, self.backoff = workers, retries, backoff
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def _chat(self, images: list[np.ndarray], text: str, max_tokens: int, logprobs: bool) -> dict:
        content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}} for im in images]
        content.append({"type": "text", "text": text})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0.0}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=10)
        last: str = ""
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.TransportError as e:  # includes timeouts
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                elif r.status_code >= 400:
                    raise VLMError(f"VLM rejected request: HTTP {r.status_code}: {r.text[:200]}")
                else:
                    try:
                        return r.json()
                    except ValueError as e:
                        raise VLMError(f"malformed VLM response: {e}") from e
            if attempt < self.retries:
                time.sleep(self.backoff * (attempt + 1))
        raise VLMError(f"VLM unavailable after {self.retries + 1} attempts: {last}")

    def yes_prob(self, images: list[np.ndarray], question: str) -> float:
        resp = self._chat(images, question + YES_NO_SUFFIX, 1, True)
        try:
            choice = resp["choices"][0]
            content = (choice.get("logprobs") or {}).get("content") or []
            if not content:
                global _warned_no_logprobs
                if not _warned_no_logprobs:
                    _warned_no_logprobs = True
                    log.warning("VLM server returned no logprobs; P(yes) degrades to 0/1 from the reply text")
                return 1.0 if choice["message"]["content"].strip().lower().startswith("yes") else 0.0
            return p_yes(content[0]["top_logprobs"])
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise VLMError(f"malformed VLM response: {e!r}") from e

    def yes_probs(self, items: list[tuple[list[np.ndarray], str]]) -> list[float]:
        abort = threading.Event()

        def run(item: tuple[list[np.ndarray], str]) -> float:
            if abort.is_set():
                raise VLMError("aborted after an earlier VLM failure")
            try:
                return self.yes_prob(*item)
            except VLMError:
                abort.set()
                raise

        ex = ThreadPoolExecutor(self.workers)
        try:
            futures = [ex.submit(run, it) for it in items]
            return [f.result() for f in futures]
        except VLMError:
            abort.set()
            raise
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def text(self, images: list[np.ndarray], prompt: str, max_tokens: int = 256) -> str:
        resp = self._chat(images, prompt, max_tokens, False)
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMError(f"malformed VLM response: {e!r}") from e


class FakeVLM:
    def __init__(self, answer: Callable[[list[np.ndarray], str], float] = lambda i, q: 0.0,
                 reply: Callable[[list[np.ndarray], str], str] = lambda i, p: ""):
        self.answer, self.reply, self.calls = answer, reply, 0

    def yes_prob(self, images, question):
        self.calls += 1
        return float(self.answer(images, question))

    def yes_probs(self, items):
        return [self.yes_prob(i, q) for i, q in items]

    def text(self, images, prompt, max_tokens=256):
        self.calls += 1
        return self.reply(images, prompt)
