"""Fetch the KESTREL pretraining and instruction data: a STEM/reasoning diet.

The 120M recipe optimized for fluent conversation. This one optimizes for
technical depth, so the mix shifts hard toward code, math, systems and science
and away from general web prose. Data choice decides what a 250M model knows;
architecture barely moves it.

Pretraining sources, and why each earns its share:

  fineweb-edu   Educational-quality web text. The language backbone -- without
                enough ordinary prose the model reads like a stack trace.
  cosmopedia-v2 Synthetic textbooks. The Phi line showed curated explanatory
                prose punches above its token count, and it is specifically what
                teaches the *explaining* voice this model is asked for.
  finemath      Filtered mathematical web text. Math is the cheapest available
                source of multi-step reasoning structure.
  starcoder     Source code, restricted to a few high-signal languages. All 300
                languages of The Stack would spend a small model's capacity on
                syntax it will never be asked for.
  pes2o         Open-access scientific papers: the "deep knowledge of science"
                half, and the only source here written by domain experts.
  wikipedia     Factual grounding and the connective tissue between fields.
  systems       Linux kernel Documentation, man pages, RFCs, the MITRE CWE and
                ATT&CK catalogues, and CVE descriptions -- public-domain or
                freely redistributable primary sources for operating systems,
                networking and security. No general web dataset covers kernel,
                protocol, syscall and vulnerability text at usable density.

SFT sources cover chat, code, math and short-form reasoning. Long chain-of-thought
is deliberately excluded: at 250M the model cannot hold a 2,000-token trace and
merely learns to imitate the shape of reasoning it cannot perform.

Nothing here is an alignment or refusal set. Both variants share this data; the
guarded variant adds its refusal training afterwards via `geocentric align-safety`,
which keeps the two builds one stage apart instead of two datasets apart.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DOC_SEP = "\n\n\n"

# Shares sum to 1.0 across the pretraining token budget.
PRETRAIN_SOURCES = {
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
        "split": "train", "field": "text", "share": 0.17,
    },
    "cosmopedia": {
        "repo": "HuggingFaceTB/cosmopedia-v2", "config": "cosmopedia-v2",
        "split": "train", "field": "text", "share": 0.18,
    },
    "finemath": {
        "repo": "HuggingFaceTB/finemath", "config": "finemath-3plus",
        "split": "train", "field": "text", "share": 0.15,
    },
    "code": {
        "repo": "bigcode/the-stack-smol-xl", "config": None,
        "split": "train", "field": "content", "share": 0.25,
        # data_dir is set per language below; the loader iterates them in turn.
        # smol-xl holds roughly 100 MB per language, so nine of them yielded
        # 935 MB against a 4.9 GB slice -- code would have been 4.8% of the
        # corpus instead of 25%. Breadth is how an ungated source reaches the
        # target. Proof assistants and dead esoterica stay excluded; what is here
        # is what a systems, security and application model actually meets.
        "languages": [
            # systems and low level
            "c", "c++", "c-sharp", "rust", "go", "assembly", "zig", "cuda",
            "fortran", "ada", "pascal", "glsl",
            # ops, shells and build
            "python", "shell", "perl", "lua", "powershell", "batchfile", "tcl",
            "awk", "makefile", "cmake", "dockerfile",
            # application and web
            "javascript", "typescript", "java", "php", "ruby", "kotlin",
            "scala", "dart", "groovy", "css", "html",
            # data, query and interchange
            "sql", "r", "matlab", "julia", "protocol-buffer", "thrift",
            # functional
            "haskell", "ocaml", "erlang", "elixir", "clojure", "common-lisp",
            "f-sharp", "scheme",
            # hardware description
            "verilog", "vhdl", "systemverilog",
            # documentation and contracts
            "markdown", "restructuredtext", "tex", "solidity", "visual-basic",
        ],
    },
    "science": {
        # allenai/peS2o is a script-based dataset and no longer loads at all.
        # common-pile serves the same corpus as parquet, and as full paper text.
        "repo": "common-pile/peS2o_filtered", "config": None,
        "split": "train", "field": "text", "share": 0.08,
    },
    "wikipedia": {
        "repo": "wikimedia/wikipedia", "config": "20231101.en",
        "split": "train", "field": "text", "share": 0.07,
    },
    "systems": {
        # Assembled from primary sources rather than a single HF repo; see
        # fetch_systems(). Kernel docs, man pages, RFCs, CWE, ATT&CK and CVEs
        # have no single HF mirror worth trusting.
        "repo": None, "share": 0.10,
    },
}

SFT_SOURCES = {
    "chat": {
        "repo": "HuggingFaceTB/smol-smoltalk", "config": None, "split": "train",
        "share": 0.30, "format": "messages",
    },
    "general": {
        "repo": "teknium/OpenHermes-2.5", "config": None, "split": "train",
        "share": 0.20, "format": "conversations",
    },
    "code_oss": {
        "repo": "ise-uiuc/Magicoder-OSS-Instruct-75K", "config": None, "split": "train",
        "share": 0.15, "format": "problem_solution",
    },
    "code_evol": {
        "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K", "config": None, "split": "train",
        "share": 0.10, "format": "instruction_response",
    },
    "math": {
        "repo": "meta-math/MetaMathQA", "config": None, "split": "train",
        "share": 0.15, "format": "query_response",
    },
    "reasoning": {
        # Default config is ShareGPT-shaped (system + conversations), not the
        # problem/solution pair the "metadata" config suggests. Its own system
        # prompt asks for long <|begin_of_thought|> traces; those blow past
        # sft_max_chars and get dropped, which is the intent at this scale --
        # what survives is the short-form reasoning a 250M model can hold.
        "repo": "open-thoughts/OpenThoughts-114k", "config": None, "split": "train",
        "share": 0.10, "format": "conversations",
    },
}

# Public instruction sets carry refusals emitted by whatever model generated them.
# Left in, they teach refusal behaviour to *both* builds, which defeats having a
# separate guarded variant at all -- the guarded one is supposed to acquire that
# behaviour deliberately, in a stage that can be measured and turned off.
REFUSAL = re.compile(
    r"^\s*(?:i(?:'m| am)\s+(?:sorry|afraid|unable)"
    r"|(?:sorry|unfortunately),?\s+(?:but\s+)?i"
    r"|i\s+(?:cannot|can't|won't|will not)\s+(?:assist|help|provide|comply|fulfill|create|write)"
    r"|as an ai(?:\s+language)?\s+model,?\s+i"
    r"|i\s+must\s+(?:decline|refuse))",
    re.IGNORECASE,
)

# A short answer that merely *mentions* being unable is fine; a short answer that
# opens with a refusal and goes nowhere is the pattern worth dropping.
MAX_REFUSAL_CHARS = 600


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def is_refusal(text: str) -> bool:
    stripped = text.strip()
    return bool(REFUSAL.match(stripped)) and len(stripped) < MAX_REFUSAL_CHARS


def stream_to_text(name: str, spec: dict, target_bytes: int, out_path: Path, resume: bool) -> int:
    from datasets import load_dataset

    if resume and out_path.exists() and out_path.stat().st_size >= target_bytes:
        print(f"[{name}] already have {human(out_path.stat().st_size)} — skipping")
        return out_path.stat().st_size

    languages = spec.get("languages") or [None]
    per_language = target_bytes // len(languages)
    print(f"[{name}] streaming {spec['repo']} -> {out_path} (target {human(target_bytes)})")

    written = 0
    documents = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for language in languages:
            kwargs = {"split": spec["split"], "streaming": True}
            if spec["config"]:
                kwargs["name"] = spec["config"]
            if language:
                kwargs["data_dir"] = f"data/{language}"
            budget = written + per_language if language else target_bytes
            try:
                stream = load_dataset(spec["repo"], **kwargs)
            except Exception as exc:
                # A renamed language directory or a dataset that changed loaders
                # must cost its own slice, not the whole run. This is a nine-day
                # pipeline; it does not get to die on one bad repo path.
                print(f"[{name}] {language or spec['repo']} unavailable, skipping: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            for row in stream:
                text = (row.get(spec["field"]) or "").strip()
                if len(text) < 200:
                    continue
                sink.write(text)
                sink.write(DOC_SEP)
                written += len(text.encode("utf-8")) + len(DOC_SEP)
                documents += 1
                if documents % 20_000 == 0:
                    print(f"[{name}] {documents:,} docs | {human(written)} | "
                          f"{100 * written / target_bytes:5.1f}%", flush=True)
                if written >= budget:
                    break
            if written >= target_bytes:
                break
    print(f"[{name}] done: {documents:,} documents, {human(written)}")
    return written


def _man_pages(sink, budget: int) -> int:
    """Every man page on this box: syscalls (2), libc (3), admin (8), and the rest.

    The operating system the model is meant to know, described by the OS itself.
    """
    import subprocess
    print("[systems] collecting local man pages")
    written = 0
    try:
        pages = subprocess.run(["apropos", "."], capture_output=True, text=True,
                               timeout=60).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pages = []
    for line in pages:
        name = line.split(" ", 1)[0]
        if not name or "(" in name:
            continue
        try:
            body = subprocess.run(["man", name], capture_output=True, text=True, timeout=10,
                                  env={"MANWIDTH": "80", "PATH": "/usr/bin:/bin"}).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        body = re.sub(r".\x08", "", body).strip()  # strip backspace-overstrike bolding
        if len(body) < 400:
            continue
        sink.write(body + DOC_SEP)
        written += len(body.encode("utf-8")) + len(DOC_SEP)
        if written >= budget:
            break
    print(f"[systems] man pages: {human(written)}")
    return written


def _kernel_docs(sink, budget: int) -> int:
    """The Linux kernel's own Documentation/ tree — GPLv2, the authoritative

    source on the kernel: memory management, scheduling, filesystems, the driver
    model, locking, namespaces, seccomp. A general web crawl has none of this at
    depth. One release tarball is fetched and only Documentation/*.rst|.txt is
    read out of the stream; the source tree itself is discarded.
    """
    import io
    import lzma
    import tarfile
    import urllib.request

    url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.tar.xz"
    print(f"[systems] fetching kernel Documentation/ from {url}")
    written = 0
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            # Decompress and untar as a stream so the ~140 MB tarball is never
            # held whole and the source tree is never written to disk.
            xz = lzma.LZMAFile(response)
            with tarfile.open(fileobj=xz, mode="r|") as tar:
                for member in tar:
                    if written >= budget:
                        break
                    if not member.isfile():
                        continue
                    parts = member.name.split("/", 1)
                    if len(parts) < 2 or not parts[1].startswith("Documentation/"):
                        continue
                    if not member.name.endswith((".rst", ".txt")):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    body = extracted.read().decode("utf-8", "replace").strip()
                    if len(body) < 400:
                        continue
                    sink.write(body + DOC_SEP)
                    written += len(body.encode("utf-8")) + len(DOC_SEP)
    except Exception as exc:
        print(f"[systems] kernel docs unavailable ({exc})")
    print(f"[systems] kernel docs: {human(written)}")
    return written


def _rfcs(sink, budget: int) -> int:
    """RFCs: how the wires actually work, from the RFC Editor. Public domain."""
    import urllib.request
    written = 0
    misses = 0
    for number in range(1, 9600):
        if written >= budget:
            break
        # Gaps in the numbering are normal (never-published numbers), so a single
        # miss means nothing -- but a long unbroken run of them means the server
        # is refusing us, and 9,600 timeouts at 15s each is 40 hours of nothing.
        if misses >= 150:
            print(f"[systems] rfc: {misses} consecutive failures, stopping early", flush=True)
            break
        try:
            with urllib.request.urlopen(
                    f"https://www.rfc-editor.org/rfc/rfc{number}.txt", timeout=15) as response:
                body = response.read().decode("utf-8", "replace").strip()
            misses = 0
        except Exception:
            misses += 1
            continue
        if len(body) < 2000:
            continue
        sink.write(body + DOC_SEP)
        written += len(body.encode("utf-8")) + len(DOC_SEP)
        if number % 100 == 0:
            print(f"[systems] rfc {number} | {human(written)}", flush=True)
    print(f"[systems] RFCs: {human(written)}")
    return written


def _cwe(sink, budget: int) -> int:
    """The MITRE CWE catalogue: weakness classes and how they arise. Freely

    redistributable with attribution. This is the vocabulary of what goes wrong
    -- buffer overflows, use-after-free, injection, race conditions -- described
    structurally rather than as one-off incidents.
    """
    import io
    import urllib.request
    import xml.etree.ElementTree as ET
    import zipfile

    print("[systems] fetching the MITRE CWE catalogue")
    written = 0
    try:
        with urllib.request.urlopen("https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
                                    timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw = archive.read(archive.namelist()[0])
        root = ET.fromstring(raw)
        ns = {"c": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        finder = ".//c:Weakness" if ns else ".//Weakness"
        for weakness in root.iterfind(finder, ns):
            name = weakness.get("Name", "")
            parts = [f"CWE-{weakness.get('ID', '')}: {name}"]
            for tag in ("Description", "Extended_Description"):
                node = weakness.find(f"c:{tag}", ns) if ns else weakness.find(tag)
                if node is not None and "".join(node.itertext()).strip():
                    parts.append("".join(node.itertext()).strip())
            record = "\n\n".join(parts)
            if len(record) < 200:
                continue
            sink.write(record + DOC_SEP)
            written += len(record.encode("utf-8")) + len(DOC_SEP)
            if written >= budget:
                break
    except Exception as exc:
        print(f"[systems] CWE unavailable ({exc})")
    print(f"[systems] CWE: {human(written)}")
    return written


def _attack(sink, budget: int) -> int:
    """MITRE ATT&CK technique descriptions (enterprise STIX). Redistributable

    with attribution. Adversary tradecraft described by the standard catalogue of
    it: how intrusions actually proceed, tactic by tactic. Descriptions only.
    """
    import urllib.request

    print("[systems] fetching MITRE ATT&CK technique descriptions")
    written = 0
    url = ("https://raw.githubusercontent.com/mitre/cti/master/"
           "enterprise-attack/enterprise-attack.json")
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            bundle = json.loads(response.read())
        for obj in bundle.get("objects", []):
            if obj.get("type") != "attack-pattern" or obj.get("x_mitre_deprecated"):
                continue
            text = (obj.get("description") or "").strip()
            if len(text) < 200:
                continue
            record = f"{obj.get('name', '')}\n\n{text}"
            sink.write(record + DOC_SEP)
            written += len(record.encode("utf-8")) + len(DOC_SEP)
            if written >= budget:
                break
    except Exception as exc:
        print(f"[systems] ATT&CK unavailable ({exc})")
    print(f"[systems] ATT&CK: {human(written)}")
    return written


def _cves(sink, budget: int) -> int:
    """NVD CVE descriptions, all years. Public domain (U.S. government work).

    Vulnerability classes described by the people who catalogue them.
    Descriptions only -- no exploit code is fetched.
    """
    import gzip
    import urllib.request

    print("[systems] fetching CVE descriptions from the NVD feeds")
    written = 0
    for year in range(2002, 2027):
        if written >= budget:
            break
        url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.loads(gzip.decompress(response.read()))
        except Exception as exc:
            print(f"[systems] cve {year} unavailable ({exc})")
            continue
        for item in payload.get("CVE_Items", []):
            cve = item.get("cve", {})
            identifier = cve.get("CVE_data_meta", {}).get("ID", "")
            notes = cve.get("description", {}).get("description_data", [])
            text = " ".join(n.get("value", "") for n in notes).strip()
            if len(text) < 200 or text.startswith("**"):  # ** REJECT ** placeholder rows
                continue
            record = f"{identifier}\n\n{text}"
            sink.write(record + DOC_SEP)
            written += len(record.encode("utf-8")) + len(DOC_SEP)
        print(f"[systems] cve {year} | {human(written)}", flush=True)
    print(f"[systems] CVEs: {human(written)}")
    return written


def fetch_systems(target_bytes: int, out_path: Path, resume: bool) -> int:
    """The systems corpus: Linux, operating systems, networking and security,

    assembled from primary sources because no general web dataset carries kernel,
    protocol, syscall and vulnerability text at usable density -- it is exactly
    the material a crawl represents worst. Every source here is public domain or
    freely redistributable (kernel docs GPLv2; CWE/ATT&CK with attribution).

    Each sub-source gets a share of the budget and degrades independently: if one
    endpoint is unreachable, its slice is simply smaller and the rest proceed.
    """
    if resume and out_path.exists() and out_path.stat().st_size >= target_bytes:
        print(f"[systems] already have {human(out_path.stat().st_size)} — skipping")
        return out_path.stat().st_size

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Shares within the systems slice. Kernel docs and man pages are the OS core;
    # RFCs the networking core; CWE/ATT&CK/CVE the security core.
    plan = [
        (_man_pages, 0.18),
        (_kernel_docs, 0.22),
        (_rfcs, 0.25),
        (_cwe, 0.05),
        (_attack, 0.05),
        (_cves, 0.25),
    ]
    written = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for fetch, share in plan:
            # Each source fills up to its own slice; an under-filled slice (an
            # unreachable endpoint, a small local man set) just makes the systems
            # corpus smaller rather than being redistributed. Honest over clever.
            written += fetch(sink, int(target_bytes * share))
    print(f"[systems] done: {human(written)}")
    return written


def normalize(row: dict, layout: str) -> list[dict] | None:
    """Reduce every upstream schema to a single messages list."""
    if layout == "messages":
        messages = row.get("messages")
    elif layout == "conversations":
        roles = {"human": "user", "gpt": "assistant", "system": "system",
                 "user": "user", "assistant": "assistant"}
        messages = [{"role": roles.get(m.get("from"), m.get("from")), "content": m.get("value")}
                    for m in (row.get("conversations") or [])]
    elif layout == "problem_solution":
        messages = [{"role": "user", "content": row.get("problem") or row.get("instruction")},
                    {"role": "assistant", "content": row.get("solution") or row.get("response")}]
    elif layout == "instruction_response":
        messages = [{"role": "user", "content": row.get("instruction")},
                    {"role": "assistant", "content": row.get("response")}]
    elif layout == "query_response":
        messages = [{"role": "user", "content": row.get("query")},
                    {"role": "assistant", "content": row.get("response")}]
    else:
        raise ValueError(f"Unknown layout {layout}")

    if not isinstance(messages, list) or len(messages) < 2:
        return None
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            return None
        cleaned.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in cleaned):
        return None
    return cleaned


def download_sft(out_path: Path, total_rows: int, resume: bool, keep_refusals: bool,
                 max_chars: int) -> int:
    from datasets import load_dataset

    if resume and out_path.exists() and out_path.stat().st_size > 0:
        rows = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"[sft] already have {rows:,} conversations — skipping")
        return rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = total = dropped_refusal = dropped_long = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for name, spec in SFT_SOURCES.items():
            budget = int(total_rows * spec["share"])
            print(f"[sft:{name}] streaming {spec['repo']} (target {budget:,} rows)")
            kwargs = {"split": spec["split"], "streaming": True}
            if spec["config"]:
                kwargs["name"] = spec["config"]
            taken = 0
            try:
                dataset = load_dataset(spec["repo"], **kwargs)
            except Exception as exc:
                print(f"[sft:{name}] unavailable, skipping: {exc}")
                continue
            for row in dataset:
                total += 1
                messages = normalize(row, spec["format"])
                if messages is None:
                    continue
                answers = [m["content"] for m in messages if m["role"] == "assistant"]
                if not keep_refusals and any(is_refusal(a) for a in answers):
                    dropped_refusal += 1
                    continue
                # A conversation longer than the context is dropped by the trainer
                # anyway; dropping it here keeps the shares honest.
                if sum(len(m["content"]) for m in messages) > max_chars:
                    dropped_long += 1
                    continue
                sink.write(json.dumps({"messages": messages, "source": name},
                                      ensure_ascii=False) + "\n")
                kept += 1
                taken += 1
                if taken % 20_000 == 0:
                    print(f"[sft:{name}] {taken:,}/{budget:,}", flush=True)
                if taken >= budget:
                    break
            print(f"[sft:{name}] kept {taken:,}")

    print(f"\n[sft] {kept:,} conversations from {total:,} rows examined")
    print(f"[sft] dropped {dropped_refusal:,} refusals, {dropped_long:,} over-length")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/kestrel")
    ap.add_argument("--target_tokens", type=float, default=5.0e9,
                    help="Pretraining token budget. Text is fetched at ~4.2 bytes/token.")
    ap.add_argument("--bytes_per_token", type=float, default=4.2)
    ap.add_argument("--sft_rows", type=int, default=600_000)
    ap.add_argument("--sft_max_chars", type=int, default=6000,
                    help="Drop conversations longer than this; ~1024 tokens of context.")
    ap.add_argument("--keep_refusals", action="store_true",
                    help="Keep canned refusals. Off by default: the guarded variant "
                         "gets its refusal behaviour from align-safety instead.")
    ap.add_argument("--only", default=None,
                    help="Fetch one of: " + ", ".join(PRETRAIN_SOURCES) + ", sft")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan and the disk it needs, download nothing.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    total_bytes = int(args.target_tokens * args.bytes_per_token)
    print(f"KESTREL data plan: {args.target_tokens / 1e9:.2f}B tokens "
          f"≈ {human(total_bytes)} of text\n")
    print(f"  {'source':<12} {'share':>6}  {'text':>10}   repo")
    for name, spec in PRETRAIN_SOURCES.items():
        repo = spec["repo"] or "kernel docs / man / RFC / CWE / ATT&CK / CVE"
        print(f"  {name:<12} {spec['share']:>5.0%}  {human(total_bytes * spec['share']):>10}   {repo}")
    print(f"\n  SFT: {args.sft_rows:,} conversations")
    for name, spec in SFT_SOURCES.items():
        print(f"  {name:<12} {spec['share']:>5.0%}  {int(args.sft_rows * spec['share']):>10,}   {spec['repo']}")
    print(f"\n  refusal filtering: {'off' if args.keep_refusals else 'on'}")
    print(f"  raw text {human(total_bytes)} + token shards ~{human(args.target_tokens * 2)}")

    if args.dry_run:
        print("\nDry run: nothing downloaded.")
        return

    resume = not args.no_resume
    for name, spec in PRETRAIN_SOURCES.items():
        if args.only and args.only != name:
            continue
        if args.only is None and name == "systems":
            pass
        target = int(total_bytes * spec["share"])
        path = out / "pretrain" / f"{name}.txt"
        try:
            if name == "systems":
                fetch_systems(target, path, resume)
            else:
                stream_to_text(name, spec, target, path, resume)
        except Exception as exc:
            # Report and carry on. A short corpus is recoverable by rerunning
            # this script; a pipeline that died overnight at source four is not.
            print(f"[{name}] FAILED, continuing without it: "
                  f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)

    if args.only in (None, "sft"):
        download_sft(out / "sft" / "kestrel_sft.jsonl", args.sft_rows, resume,
                     args.keep_refusals, args.sft_max_chars)

    print(f"\nReady. Pretraining text: {out / 'pretrain'}    SFT: {out / 'sft'}")


if __name__ == "__main__":
    main()
