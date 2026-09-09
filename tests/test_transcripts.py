import datetime
import json

from geocentric.transcripts import TranscriptStore


def test_only_consented_rows_are_offered_for_training(tmp_path):
    store = TranscriptStore(tmp_path)
    store.record("allowed", "reply", True, "r1")
    store.record("withheld", "reply", False, "r2")
    assert [row["prompt"] for row in store.training_rows()] == ["allowed"]


def test_retention_deletes_on_the_thirtieth_day(tmp_path):
    store = TranscriptStore(tmp_path, retention_days=30)
    now = datetime.datetime.now(datetime.timezone.utc)
    for age in (29, 30, 31):
        store._path(now - datetime.timedelta(days=age)).write_text("{}\n", encoding="utf-8")
    store.purge()
    survivors = sorted(p.stem for p in tmp_path.glob("*.jsonl"))
    assert survivors == [f"{now - datetime.timedelta(days=29):%Y-%m-%d}"]


def test_record_survives_an_unwritable_directory(tmp_path):
    store = TranscriptStore(tmp_path / "store")
    store.dir = tmp_path / "gone"  # never created
    store.record("prompt", "reply", True, "r1")  # must not raise into the request


def test_rows_round_trip_as_jsonl(tmp_path):
    store = TranscriptStore(tmp_path)
    store.record("hello", "world", True, "r1", {"generated_tokens": 7})
    line = next(tmp_path.glob("*.jsonl")).read_text().strip()
    row = json.loads(line)
    assert row["prompt"] == "hello"
    assert row["response"] == "world"
    assert row["generated_tokens"] == 7
    assert row["training_consent"] is True
