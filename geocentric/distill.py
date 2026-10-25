"""Distill training data from a local Ollama model into SFT-ready JSONL.

A teacher model served by Ollama answers a list of prompts; each answer is written as a
``messages`` record that ``geocentric sft --sft_data_path`` reads directly. Only the
standard library is used, so no extra dependency is needed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Iterator

PREFERRED_MODEL = "deepseek-coder-v2:16b"
DEFAULT_HOST = "http://localhost:11434"
OUTPUT_NAME = "distilled.jsonl"

SEED_PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "How do I read a CSV file and compute the average of a column in Python?",
    "Explain what a Python decorator is and show a small example.",
    "Write a function that returns the n-th Fibonacci number without recursion.",
    "What is the difference between a process and a thread?",
    "Explain Big-O notation with examples of O(1), O(n) and O(n^2).",
    "Write a JavaScript function that debounces another function.",
    "Explain how self-attention works in a transformer in plain language.",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "Explain what a hash table is and why lookups are fast on average.",
    "Write a Python class implementing a stack with push, pop and peek.",
    "How does gradient descent minimise a loss function?",
    "Write a bash one-liner that counts the lines in every .py file in a folder.",
    "Explain what a race condition is and how a lock prevents one.",
]


class DistillError(RuntimeError):
    pass


def _request(host: str, path: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise DistillError(f"Ollama returned HTTP {exc.code} for {path}: {exc.read().decode('utf-8', 'replace')}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DistillError(f"Cannot reach Ollama at {host} ({exc}). Is `ollama serve` running?") from exc


def installed_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[str]:
    return [m["name"] for m in _request(host, "/api/tags", None, timeout).get("models", [])]


def resolve_model(requested: str | None, installed: list[str]) -> str:
    """Pick the teacher. deepseek-coder-v2:16b wins unless another model was asked for by name."""
    if requested:
        if requested not in installed and f"{requested}:latest" not in installed:
            raise DistillError(f"Model {requested!r} is not installed. Run `ollama pull {requested}`.")
        return requested
    if PREFERRED_MODEL in installed:
        return PREFERRED_MODEL
    raise DistillError(
        f"Preferred teacher {PREFERRED_MODEL} is not installed. Run `ollama pull {PREFERRED_MODEL}`, "
        f"or pass --model with one of: {', '.join(installed) or 'none installed'}."
    )


def load_prompts(path: str | Path | None) -> list[str]:
    """Read prompts from .txt (one per line) or .jsonl ({"prompt"|"instruction"|"question": ...})."""
    if path is None:
        return list(SEED_PROMPTS)
    path = Path(path)
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if path.suffix.lower() == ".jsonl":
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            line = next((str(row[k]).strip() for k in ("prompt", "instruction", "question") if row.get(k)), "")
        if line:
            prompts.append(line)
    return list(dict.fromkeys(prompts))  # drop duplicates, keep order


def _done_prompts(out_file: Path) -> set[str]:
    done: set[str] = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            try:
                msgs = json.loads(line)["messages"]
                done.update(m["content"] for m in msgs if m["role"] == "user")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def default_chat(host: str, model: str, temperature: float, max_tokens: int, timeout: float) -> Callable[[str, str | None], str]:
    def chat(prompt: str, system: str | None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = _request(host, "/api/chat", {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout)
        return (body.get("message") or {}).get("content", "")
    return chat


def distill(
    output_dir: str | Path,
    prompts: Iterable[str],
    chat: Callable[[str, str | None], str],
    system: str | None = None,
    teacher: str = PREFERRED_MODEL,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[dict]:
    """Append one record per prompt to ``<output_dir>/distilled.jsonl``; prompts already there are skipped.

    Each record is flushed as written, so an interrupted run resumes where it stopped.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / OUTPUT_NAME
    pending = [p for p in prompts if p not in _done_prompts(out_file)]
    with out_file.open("a", encoding="utf-8") as fh:
        for i, prompt in enumerate(pending, 1):
            reply = chat(prompt, system).strip()
            if on_progress:
                on_progress(i, len(pending), prompt)
            if not reply:
                continue
            messages = ([{"role": "system", "content": system}] if system else [])
            messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
            record = {"messages": messages, "teacher": teacher}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            yield record
