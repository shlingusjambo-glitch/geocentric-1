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


def test_tester_rows_are_marked_and_never_trainable(tmp_path):
    from geocentric.testers import TesterRegistry, hash_key

    store = TranscriptStore(tmp_path)
    # The server passes `training_consent and not tester`, so even a consenting
    # authorised tester never lands in a training set.
    store.record("probe", "reply", False, "r1", tester="under-16 safety review")
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text().strip())
    assert row["tester"] == "under-16 safety review"
    assert row["training_consent"] is False
    assert list(store.training_rows()) == []

    keys = tmp_path / "testers.txt"
    keys.write_text(f"{hash_key('let-me-in')}  riley\n", encoding="utf-8")
    registry = TesterRegistry(keys)
    assert registry.verify("let-me-in") == "riley"
    assert registry.verify("nope") is None


def test_reports_are_stored_separately_and_expire(tmp_path):
    import datetime

    store = TranscriptStore(tmp_path, retention_days=30)
    assert store.record_report({"what_went_wrong": "invented a date", "messages": []})
    report_file = next(tmp_path.glob("report-*.jsonl"))
    row = json.loads(report_file.read_text().strip())
    assert row["what_went_wrong"] == "invented a date"
    assert "at" in row
    # Reports are not chat rows, so they never reach a training set.
    assert list(store.training_rows()) == []

    stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=31)
    (tmp_path / f"report-{stale:%Y-%m-%d}.jsonl").write_text("{}\n", encoding="utf-8")
    store.purge()
    assert not (tmp_path / f"report-{stale:%Y-%m-%d}.jsonl").exists()
    assert report_file.exists()


def test_load_state_and_pacing_are_reported_to_the_client(tmp_path):
    from geocentric.load import LoadMonitor, PRESSURE_EVENTS, paced

    now = [0.0]
    monitor = LoadMonitor(clock=lambda: now[0], thermal=lambda: False)
    assert monitor.state()["tokens_per_second"] == 25
    for _ in range(PRESSURE_EVENTS):
        monitor.note_contention()
    high = monitor.state()
    assert high["load"] == "high"
    assert high["tokens_per_second"] == 10
    # The generation counter is what lets a client that was away tell the
    # difference between "still high" and "high again".
    assert monitor.state()["generation"] == high["generation"]

    fake = [0.0]

    def sleep(seconds):
        fake[0] += seconds

    out = list(paced(iter("abcde"), 5, sleep=sleep, clock=lambda: fake[0]))
    assert out == list("abcde")
    assert abs(fake[0] - 1.0) < 0.01
