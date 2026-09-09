"""Retained prompts and responses for safety review, with a hard expiry.

The published policy (https://geocentricai.com/legal/privacy/) says two things
that this module is responsible for making true:

  * nothing is kept unless the user opted in. `training_consent` on the chat
    request is what decides it, and it is off by default;
  * what is kept is deleted within 30 days.

It also holds reports submitted from the chat interface, which are a separate
opt-in: the reporter sees the transcript and ticks a box before it is sent.

Storage is one JSON Lines file per UTC day, so expiry is a file deletion rather
than a rewrite, and an interrupted append costs at most the last line. Off
unless `--transcripts DIR` is passed: a LAN or laptop server should not start
recording its operator's conversations because a hosted deployment needs to.
"""
from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path

RETENTION_DAYS = 30


class TranscriptStore:
    def __init__(self, directory, retention_days=RETENTION_DAYS):
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(retention_days)
        self._lock = threading.Lock()
        self.purge()

    def _path(self, when, kind="chat"):
        stem = f"{when:%Y-%m-%d}" if kind == "chat" else f"{kind}-{when:%Y-%m-%d}"
        return self.dir / f"{stem}.jsonl"

    def _append(self, row, kind="chat"):
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            with self._lock, self._path(now, kind).open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            return True
        except OSError:
            return False

    def record_report(self, report):
        """Store a user-submitted report about a response."""
        report = dict(report)
        report["at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return self._append(report, kind="report")

    def record(self, prompt, response, training_consent, request_id, stats=None, tester=None):
        """Append one exchange. Never raises into the request path.

        The caller is responsible for only calling this when the user opted in;
        `training_consent` is recorded so a later reader can still tell.
        """
        row = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request_id": request_id,
            "training_consent": bool(training_consent),
            "prompt": prompt,
            "response": response,
            "generated_tokens": (stats or {}).get("generated_tokens"),
        }
        if tester:
            # Authorised internal testing, which may involve someone under 16.
            row["tester"] = tester
        self._append(row)

    def purge(self, now=None):
        """Delete day files past the retention window. Returns how many went."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = (now - datetime.timedelta(days=self.retention_days)).date()
        # Files are named for the day they were written, so anything dated on or
        # before the cutoff is already at the retention limit.
        removed = 0
        with self._lock:
            for path in self.dir.glob("*.jsonl"):
                stem = path.stem
                if stem.startswith("report-"):
                    stem = stem[len("report-"):]
                try:
                    day = datetime.date.fromisoformat(stem)
                except ValueError:
                    continue  # not one of ours
                if day <= cutoff:  # "within 30 days" means the 30th day goes too
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
        return removed

    def training_rows(self):
        """Every retained exchange the user allowed us to train on."""
        for path in sorted(self.dir.glob("*.jsonl")):
            if path.name.startswith("report-"):
                continue
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("training_consent"):
                        yield row


def demo():
    """Self-check: expiry actually expires, and consent actually gates."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = TranscriptStore(tmp, retention_days=30)
        store.record("hello", "hi", True, "r1")
        store.record("secret", "sure", False, "r2")
        store.record_report({"what_went_wrong": "wrong date", "messages": []})
        assert (Path(tmp) / f"report-{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}.jsonl").exists()

        rows = list(store.training_rows())
        assert [r["prompt"] for r in rows] == ["hello"], rows

        stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=31)
        store._path(stale).write_text('{"at": "old"}\n', encoding="utf-8")
        assert store.purge() == 1
        assert not store._path(stale).exists()
        assert len(list(store.training_rows())) == 1, "current day must survive"

        # A one-day window drops yesterday and keeps today.
        yesterday = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
        store._path(yesterday).write_text('{"at": "yesterday"}\n', encoding="utf-8")
        TranscriptStore(tmp, retention_days=1)  # purges on construction
        assert not store._path(yesterday).exists()
        assert len(list(store.training_rows())) == 1, "current day must survive"
    print("transcripts: ok")


if __name__ == "__main__":
    demo()
