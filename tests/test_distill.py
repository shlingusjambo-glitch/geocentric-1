import json

import pytest

from geocentric.distill import (
    OUTPUT_NAME, PREFERRED_MODEL, DistillError, distill, load_prompts, resolve_model,
)


def test_preferred_model_is_chosen_by_default():
    assert resolve_model(None, ["llama3:8b", PREFERRED_MODEL]) == PREFERRED_MODEL


def test_missing_preferred_model_is_an_error_not_a_silent_swap():
    with pytest.raises(DistillError):
        resolve_model(None, ["llama3:8b"])


def test_explicit_model_overrides_preference():
    assert resolve_model("llama3:8b", ["llama3:8b", PREFERRED_MODEL]) == "llama3:8b"


def test_records_are_sft_messages_in_the_output_dir(tmp_path):
    out = tmp_path / "data"
    rows = list(distill(out, ["hi"], lambda p, s: " hello ", system="Be terse."))
    lines = (out / OUTPUT_NAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == rows
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert rows[0]["messages"][-1]["content"] == "hello"


def test_rerun_skips_finished_prompts_and_empty_replies(tmp_path):
    calls = []

    def chat(p, s):
        calls.append(p)
        return "" if p == "blank" else "ok"

    list(distill(tmp_path, ["a", "blank"], chat, system="sys"))
    list(distill(tmp_path, ["a", "b"], chat, system="sys"))
    assert calls == ["a", "blank", "b"]
    assert len((tmp_path / OUTPUT_NAME).read_text().splitlines()) == 2


def test_prompt_files(tmp_path):
    (tmp_path / "p.txt").write_text("one\n\ntwo\none\n")
    (tmp_path / "p.jsonl").write_text('{"instruction": "x"}\nnot json\n{"prompt": "y"}\n')
    assert load_prompts(tmp_path / "p.txt") == ["one", "two"]
    assert load_prompts(tmp_path / "p.jsonl") == ["x", "y"]
    assert load_prompts(None)
