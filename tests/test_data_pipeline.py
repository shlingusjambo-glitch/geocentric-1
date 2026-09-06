"""Regression tests for the failures that made the 2.1 models incoherent."""
from __future__ import annotations

import json

import numpy as np
import pytest

from geocentric.data import PackedDataset, SFTDataset, iter_documents, prepare_corpus
from geocentric.tokenizer_train import SPECIAL_TOKENS, train_byte_bpe_tokenizer


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    text = "\n".join(
        f"Country {i} is a nation. Its capital is City {i}, a place known for trade and rivers."
        for i in range(800)
    )
    (d / "corpus.txt").write_text(text, encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return train_byte_bpe_tokenizer(
        iter_documents(corpus / "corpus.txt"), corpus / "tokenizer.json", vocab_size=1500
    )


def test_text_files_are_not_split_into_lines(corpus):
    """The 2.1 loader yielded one 'document' per line and put <eos> after each."""
    docs = list(iter_documents(corpus / "corpus.txt"))
    assert len(docs) == 1
    assert docs[0].count("\n") > 100


def test_doc_sep_splits_on_real_boundaries(tmp_path):
    (tmp_path / "d.txt").write_text("alpha one\n\nbeta two\n\ngamma three", encoding="utf-8")
    assert len(list(iter_documents(tmp_path / "d.txt", doc_sep="\n\n"))) == 3


def test_tokenizer_carries_every_special_token(tokenizer):
    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) is not None


def test_eos_is_rare_not_per_line(corpus, tokenizer, tmp_path):
    out = tmp_path / "bin"
    counts = prepare_corpus(tokenizer, corpus / "corpus.txt", out, val_fraction=0.0, progress=False)
    assert counts["documents"] == 1
    tokens = np.fromfile(out / "train.bin", dtype=np.uint16)
    eos = tokenizer.token_to_id("<eos>")
    # One end-of-sequence for the whole corpus, not one every ~15 tokens.
    assert int((tokens == eos).sum()) <= 1
    assert len(tokens) > 5000


def test_packed_windows_are_shifted_by_one(corpus, tokenizer, tmp_path):
    out = tmp_path / "bin2"
    prepare_corpus(tokenizer, corpus / "corpus.txt", out, val_fraction=0.01, progress=False)
    ds = PackedDataset(out / "train.bin", block_size=64)
    assert len(ds) > 10
    item = ds[0]
    assert item["input_ids"].shape == item["labels"].shape == (64,)
    assert (item["input_ids"][1:] == item["labels"][:-1]).all()


def test_validation_split_is_a_contiguous_tail(corpus, tokenizer, tmp_path):
    out = tmp_path / "bin3"
    counts = prepare_corpus(tokenizer, corpus / "corpus.txt", out, val_fraction=0.1, progress=False)
    assert counts["val"] > 0
    total = counts["train"] + counts["val"]
    assert abs(counts["val"] / total - 0.1) < 0.02


def test_sft_trains_only_on_assistant_turns(tokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "Where is City 1"},
            {"role": "assistant", "content": "City 1 is a capital."},
            {"role": "user", "content": "And City 2"},
            {"role": "assistant", "content": "City 2 is also a capital."},
        ]
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    ds = SFTDataset(tokenizer, path, block_size=256)
    labels = ds[0]["labels"].tolist()
    trained = [t for t in labels if t != -100]
    assert trained, "no supervised tokens"
    assert any(t == -100 for t in labels), "prompt was not masked"
    decoded = tokenizer.decode(trained, skip_special_tokens=False)
    assert "City 1 is a capital." in decoded
    assert "City 2 is also a capital." in decoded
    assert "Where is City 1" not in decoded


def test_sft_supervises_the_stop_token(tokenizer, tmp_path):
    """Without <|eot|> in the labels the model never learns to stop talking."""
    path = tmp_path / "sft2.jsonl"
    path.write_text(json.dumps({"instruction": "Say hi", "output": "Hello there."}), encoding="utf-8")
    ds = SFTDataset(tokenizer, path, block_size=256)
    trained = [t for t in ds[0]["labels"].tolist() if t != -100]
    assert tokenizer.token_to_id("<|eot|>") in trained


def test_overlong_conversations_are_dropped_not_truncated(tokenizer, tmp_path):
    path = tmp_path / "sft3.jsonl"
    rows = [
        {"instruction": "short", "output": "fine"},
        {"instruction": "long " * 400, "output": "also long " * 200},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = SFTDataset(tokenizer, path, block_size=64, drop_overlong=True)
    assert ds.dropped >= 1
    for i in range(len(ds)):
        assert ds[i]["input_ids"].numel() <= 64


def test_validation_split_covers_every_source(tmp_path):
    """A tail of the whole stream is whatever file sorts last.

    When html.txt joined the corpus it sorted after fineweb, so the validation set
    became 100% markup and eval loss stopped measuring language. The split must
    take a slice of each source instead.
    """
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "a_prose.txt").write_text(
        "\n\n\n".join(f"Prose document {i} about rivers and weather and trade. " * 12
                      for i in range(200)), encoding="utf-8")
    (src / "z_markup.txt").write_text(
        "\n\n\n".join(f"<html><body><p>page {i}</p></body></html> " * 12
                      for i in range(200)), encoding="utf-8")

    # A tokenizer trained on both sources, so decoding can actually round-trip each.
    tok = train_byte_bpe_tokenizer(iter_documents(src, doc_sep="\n\n\n"),
                                   tmp_path / "tok.json", vocab_size=900)
    out = tmp_path / "bin"
    counts = prepare_corpus(tok, src, out, val_fraction=0.2, doc_sep="\n\n\n", progress=False)
    assert counts["val"] > 0

    val = np.fromfile(out / "val.bin", dtype=np.uint16)
    text = tok.decode([int(x) for x in val], skip_special_tokens=True)
    assert "<html>" in text, "markup source missing from validation split"
    assert "Prose document" in text, "prose source missing from validation split"

    # And training must still hold the bulk of both.
    train = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert len(train) > len(val) * 3
