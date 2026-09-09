"""Retained prompts and responses for safety review, with a hard expiry.

The published policy (https://geocentricai.com/legal/privacy/) says two things
that this module is responsible for making true:

  * prompts and model responses are kept for at most 30 days, then deleted;
  * they are used to train models only where the user opted in, which arrives
    as `training_consent` on the chat request.

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

    def _path(self, when):
        return self.dir / f"{when:%Y-%m-%d}.jsonl"

    def record(self, prompt, response, training_consent, request_id, stats=None, tester=None):
        """Append one exchange. Never raises into the request path."""
        now = datetime.datetime.now(datetime.timezone.utc)
        row = {
            "at": now.isoformat(),
            "request_id": request_id,
            "training_consent": bool(training_consent),
            "prompt": prompt,
            "response": response,
            "generated_tokens": (stats or {}).get("generated_tokens"),
        }
        if tester:
            # Authorised internal testing, which may involve someone under 16.
            row["tester"] = tester
        try:
            with self._lock, self._path(now).open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def purge(self, now=None):
        """Delete day files past the retention window. Returns how many went."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = (now - datetime.timedelta(days=self.retention_days)).date()
        # Files are named for the day they were written, so anything dated on or
        # before the cutoff is already at the retention limit.
        removed = 0
        with self._lock:
            for path in self.dir.glob("*.jsonl"):
                try:
                    day = datetime.date.fromisoformat(path.stem)
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
