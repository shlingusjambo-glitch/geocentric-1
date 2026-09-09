"""Authorised access for internal testers, including people under 16.

The public minimum age is 16 (https://geocentricai.com/legal/childrens-privacy/).
Geocentric may separately authorise a named individual to use the service for
development, safety testing, or age-suitability evaluation — work that a person
under 16 can legitimately be paid to do, and that cannot be done without them.

Authorisation is issued by Geocentric and verified here. It is deliberately not
a checkbox the visitor can tick for themselves: a self-declared bypass of an age
gate is not a bypass, it is a gate with a hole in it.

Key file format, one record per line, `#` for comments:

    <sha256 of the key>  <label>

Generate a key and its hash with:

    python -m geocentric.testers --new "riley, safety review"
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path


def hash_key(key):
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


class TesterRegistry:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.records = self._load()

    def _load(self):
        records = {}
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, label = line.partition(" ")
            records[digest.strip().lower()] = label.strip() or "authorised tester"
        return records

    def verify(self, key):
        """Return the tester's label, or None. Compared in constant time."""
        if not isinstance(key, str) or not key.strip():
            return None
        candidate = hash_key(key)
        for digest, label in self.records.items():
            if hmac.compare_digest(candidate, digest):
                return label
        return None


def demo():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        key = secrets.token_urlsafe(24)
        path = Path(tmp) / "testers.txt"
        path.write_text(f"# internal\n{hash_key(key)}  under-16 safety review\n", encoding="utf-8")

        registry = TesterRegistry(path)
        assert registry.verify(key) == "under-16 safety review"
        assert registry.verify(key + "x") is None
        assert registry.verify("") is None
        assert registry.verify(None) is None
        assert TesterRegistry(Path(tmp) / "absent.txt").verify(key) is None
        # The plaintext key is never written to the registry file.
        assert key not in path.read_text(encoding="utf-8")
    print("testers: ok")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", metavar="LABEL", help="Mint a key and print its registry line")
    args = parser.parse_args()
    if args.new:
        key = secrets.token_urlsafe(24)
        print(f"key (give this to the tester, once):\n  {key}\n")
        print(f"registry line (append to your tester key file):\n  {hash_key(key)}  {args.new}")
    else:
        demo()
