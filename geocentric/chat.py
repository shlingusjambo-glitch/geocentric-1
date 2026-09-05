from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|eot|>"

DEFAULT_SYSTEM = "You are Geocentric, a helpful assistant. Answer the user directly and concisely."

Message = Mapping[str, str]


def normalize_messages(rows: Sequence[Message]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        if role in {"assistant", "gpt", "bot", "model"}:
            role = "assistant"
        elif role in {"system", "instruction"}:
            role = "system"
        else:
            role = "user"
        out.append({"role": role, "content": content})
    return out


def render_chat(messages: Sequence[Message], add_generation_prompt: bool = False) -> Tuple[str, List[Tuple[int, int]]]:
    """Render a conversation into the training string.

    Returns the text plus the character spans of assistant content (including the
    trailing end-of-turn token). Those spans are the only tokens SFT trains on —
    everything else is masked, so the model learns to answer rather than to
    reproduce the prompt back at the user.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0

    def emit(chunk: str) -> int:
        nonlocal cursor
        parts.append(chunk)
        start = cursor
        cursor += len(chunk)
        return start

    for msg in messages:
        role = msg["role"]
        tag = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}[role]
        emit(f"{tag}\n")
        if role == "assistant":
            start = emit(f"{msg['content']}{EOT}\n")
            spans.append((start, cursor))
        else:
            emit(f"{msg['content']}{EOT}\n")

    if add_generation_prompt:
        emit(f"{ASSISTANT}\n")

    return "".join(parts), spans


def messages_from_record(row: Mapping[str, object]) -> List[dict]:
    """Accept the common instruction-tuning record shapes and return chat messages."""
    if isinstance(row.get("messages"), list):
        return normalize_messages(row["messages"])  # type: ignore[arg-type]

    if isinstance(row.get("conversations"), list):
        rows = []
        for turn in row["conversations"]:  # type: ignore[union-attr]
            if isinstance(turn, Mapping):
                rows.append({"role": str(turn.get("from", "user")), "content": str(turn.get("value", ""))})
        return normalize_messages(rows)

    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    response = str(row.get("response", row.get("output", ""))).strip()
    system = str(row.get("system", "")).strip()

    if not instruction or not response:
        return []

    user = instruction if not input_text else f"{instruction}\n\n{input_text}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": response})
    return messages


def format_prompt(instruction: str, input_text: str = "", system: str | None = None) -> str:
    """Build an inference prompt in the same format the model was trained on."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user = instruction.strip()
    if input_text.strip():
        user = f"{user}\n\n{input_text.strip()}"
    messages.append({"role": "user", "content": user})
    text, _ = render_chat(messages, add_generation_prompt=True)
    return text
