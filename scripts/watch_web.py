"""Broadcast a training run's status over the LAN as a web page.

Same numbers as `watch_training.py`, readable from a phone. It exists so a run
that takes nine days can be checked on from somewhere other than this desk.

**It must never cost the trainer anything.** That constraint drives the design:

  * stdlib only. torch is never imported, so this process never creates a CUDA
    context and never reserves a byte of VRAM.
  * The only input is `training_metrics.json`, which the trainer already writes.
    Nothing here touches the GPU, the model, the checkpoints or the corpus.
  * `os.nice(19)` -- lowest priority. On a contended box the kernel hands the
    trainer the core first, every time.
  * The file is read at most once per `--poll` seconds and cached, so twenty
    browser tabs refreshing cost one stat() and one small read between them,
    not twenty.

It stops when training stops: once the trainer process is gone and the metrics
file reports a terminal status, the server shuts itself down rather than lingering
as an open LAN port serving a frozen number.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from pipeline_status import describe as pipeline_describe
except Exception:
    pipeline_describe = None

TERMINAL = {"complete", "stopped", "failed", "diverged"}
# Shared with watch_training.py: the trainer writes its pid here.
PID_FILE = "trainer.pid"


def lan_ip() -> str:
    """This machine's LAN address, by asking the routing table which source

    address would be used to reach the internet. No packet is actually sent --
    a UDP connect() only selects a route.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def _argv(pid: str | int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (OSError, PermissionError):
        return []
    return raw.decode("utf-8", "replace").split("\0")


def _owns_run(pid: str | int, run_dir: Path) -> bool:
    """True if this pid is a geocentric trainer working on this run directory.

    argv elements are compared exactly rather than substring-searched: a shell
    running `grep pretrain logs/pretrain.log` mentions every keyword a trainer
    does, and a substring match reports it as training.
    """
    argv = _argv(pid)
    if "geocentric.cli" not in argv or not {"pretrain", "sft", "pipeline"} & set(argv):
        return False
    try:
        cwd = (Path("/proc") / str(pid) / "cwd").resolve()
    except (OSError, PermissionError):
        cwd = Path.cwd()
    target = run_dir.resolve()
    # --output_dir/--model_dir are usually relative, so resolve each element
    # against the process's own working directory rather than string-matching.
    return any((cwd / arg).resolve() == target for arg in argv if arg and not arg.startswith("-"))


def trainer_alive(run_dir: Path) -> bool:
    """True while a process that owns this run is still running.

    The pid file is the fast path -- `pretrain` writes it -- but `sft` does not,
    so a missing or stale file falls back to scanning /proc. Getting this wrong
    in the pessimistic direction would shut the monitor down mid-run.
    """
    try:
        pid = int((run_dir / PID_FILE).read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        pass
    else:
        # A recycled pid belonging to something unrelated must not read as live.
        if not Path("/proc").is_dir() or _owns_run(pid, run_dir):
            return True

    if not Path("/proc").is_dir():
        return False
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and _owns_run(entry.name, run_dir):
            return True
    return False


def last_activity(run_dir: Path, data_dir: Path) -> float:
    """Seconds since anything the pipeline writes last changed.

    The pre-training stages have no metrics file to go stale, so freshness on
    disk is the only evidence that a download or a tokenization is still moving.
    """
    newest = 0.0
    for pattern in ((data_dir / "pretrain").glob("*.txt"),
                    (data_dir / "sft").glob("*.jsonl"),
                    (run_dir / "corpus").glob("*.bin"),
                    [run_dir / "pipeline.json", run_dir / "tokenizer.json"]):
        for path in pattern:
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                pass
    return (time.time() - newest) if newest else float("inf")


class Monitor:
    """Caches the metrics file so browser refreshes cannot amplify disk reads."""

    def __init__(self, run_dir: Path, poll: float, data_dir: Path | None = None):
        self.run_dir = run_dir
        self.data_dir = Path(data_dir) if data_dir else Path("data/kestrel")
        self.poll = poll
        self.path = run_dir / "training_metrics.json"
        self._lock = threading.Lock()
        self._cached: dict = {}
        self._read_at = 0.0
        self.started = time.time()
        self.history: list[float] = []
        self.steps: list[int] = []

    def read(self) -> dict:
        with self._lock:
            now = time.time()
            if now - self._read_at < self.poll and self._cached:
                return self._cached
            self._read_at = now
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A half-written file is a transient, not an error worth showing.
                return self._cached or {"status": "waiting", "message": "No metrics yet."}
            self._cached = data
            return data

    def snapshot(self) -> dict:
        m = self.read()
        config = m.get("config", {})
        step = m.get("step") or 0
        total = config.get("total_steps") or 0
        loss = m.get("loss")
        alive = trainer_alive(self.run_dir)
        status = m.get("status", "unknown")

        if loss is not None and (not self.history or self.history[-1] != loss):
            self.history.append(float(loss))
            self.steps.append(int(step))
            del self.history[:-240], self.steps[:-240]

        rate = m.get("tokens_per_second") or 0
        tokens_per_step = config.get("tokens_per_step") or 0
        seconds_left = 0.0
        if rate and tokens_per_step and total > step:
            seconds_left = (total - step) * tokens_per_step / rate

        stage = {}
        if pipeline_describe is not None:
            try:
                stage = pipeline_describe(self.run_dir, self.data_dir)
            except Exception:
                stage = {}

        # Before training starts there are no metrics, but the pipeline is very
        # much alive -- a download or a corpus tokenization. Treat a live
        # pre-training stage as running so the page does not read as dead.
        # A pre-training stage counts as live only while it is demonstrably
        # moving. Without that check a pipeline that died mid-download would
        # hold the page open forever, reporting a stage that stopped hours ago.
        idle = last_activity(self.run_dir, self.data_dir)
        pre_training = bool(stage) and stage.get("stage", 0) in (1, 2, 3) and idle < 1800
        pre_training_dead = (bool(stage) and stage.get("stage", 0) in (1, 2, 3)
                             and idle >= 1800)
        if pre_training:
            alive = True
            status = stage.get("stage_name", status)
        elif pre_training_dead:
            status = f"{stage.get('stage_name', 'pipeline')} stalled"

        return {
            "run": self.run_dir.name,
            "stage": stage,
            "pre_training": pre_training,
            "phase": m.get("phase", "?"),
            "status": status,
            "alive": alive,
            # A stalled pre-training stage is an ending too: nothing has been
            # written in half an hour and no trainer is running.
            "finished": ((status in TERMINAL and not alive and not pre_training)
                         or pre_training_dead),
            "idle_seconds": idle,
            "message": m.get("message", ""),
            "step": step,
            "total": total,
            "fraction": (step / total) if total else 0.0,
            "epoch": m.get("epoch"),
            "epochs": m.get("epochs"),
            "loss": loss,
            "perplexity": m.get("perplexity"),
            "eval_loss": m.get("eval_loss"),
            "lr": m.get("lr"),
            "tokens_per_second": rate,
            "tokens_seen": m.get("tokens_seen"),
            "params": config.get("params"),
            "dtype": str(config.get("dtype", "")).replace("torch.", ""),
            "eta_seconds": seconds_left,
            "eta_text": human_time(seconds_left),
            "finish_at": (datetime.now() + timedelta(seconds=seconds_left)).strftime("%a %d %b %H:%M")
                         if seconds_left else "",
            "loss_guard": m.get("loss_guard", {}),
            "last_update": m.get("last_update"),
            "stale_seconds": stale_seconds(m.get("last_update")),
            "history": self.history,
            "history_steps": self.steps,
            "served_at": datetime.now().strftime("%H:%M:%S"),
        }


def stale_seconds(stamp: str | None) -> float:
    """How long since the trainer last wrote. A rising number means a stall."""
    if not stamp:
        return -1.0
    try:
        parsed = datetime.fromisoformat(str(stamp).rstrip("Z")).replace(tzinfo=timezone.utc)
    except ValueError:
        return -1.0
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def human_time(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__RUN__ — training</title>
<style>
:root{color-scheme:dark;--bg:#0b0d10;--card:#14181d;--line:#232a32;--dim:#8b98a6;--fg:#e6edf3;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;padding:16px}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:17px;margin:0 0 2px;font-weight:600}
.sub{color:var(--dim);font-size:13px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.bar{height:12px;background:#0b0d10;border-radius:6px;overflow:hidden;border:1px solid var(--line);margin:10px 0 6px}
.fill{height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);transition:width .6s ease}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:19px;margin-top:2px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.live{background:var(--ok);box-shadow:0 0 8px var(--ok)}.idle{background:var(--dim)}.err{background:var(--bad)}
.warnrow{color:var(--warn)}.badrow{color:var(--bad)}
svg{width:100%;height:90px;display:block}
footer{color:var(--dim);font-size:12px;text-align:center;margin-top:18px}
.msg{color:var(--dim);font-size:13px;margin-top:8px;word-break:break-word}
.stages{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.st{flex:1 1 88px;padding:6px 8px;border-radius:7px;border:1px solid var(--line);background:#0b0d10;font-size:11px;color:var(--dim)}
.st b{display:block;font-size:12px;color:var(--fg);font-weight:600;margin-top:1px}
.st.done{border-color:#1a4d2a;color:var(--ok)}.st.done b{color:var(--ok)}
.st.live{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}.st.live b{color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:3px 0;color:var(--dim)}
td.n{color:var(--fg)}td.r{text-align:right;font-variant-numeric:tabular-nums}
.mini{display:inline-block;width:70px;height:6px;background:#0b0d10;border:1px solid var(--line);border-radius:3px;overflow:hidden;vertical-align:middle}
.mini i{display:block;height:100%;background:var(--accent)}
</style></head><body><div class="wrap">
<h1><span id="dot" class="dot idle"></span><span id="title">__RUN__</span></h1>
<div class="sub" id="sub">connecting…</div>

<div class="card">
  <div class="k">pipeline</div>
  <div class="stages" id="stages"></div>
  <div class="sub" id="stagedetail" style="margin:8px 0 0"></div>
</div>

<div class="card" id="dlcard" hidden>
  <div class="k">corpus download</div>
  <div class="bar"><div class="fill" id="dlfill" style="width:0%"></div></div>
  <div class="sub" id="dltotal" style="margin:0 0 8px"></div>
  <table id="dltable"></table>
</div>

<div class="card" id="trcard">
  <div class="k">progress</div>
  <div class="bar"><div class="fill" id="fill" style="width:0%"></div></div>
  <div id="prog" class="sub" style="margin:0"></div>
  <div class="msg" id="msg"></div>
</div>

<div class="card grid">
  <div><div class="k">loss</div><div class="v" id="loss">—</div></div>
  <div><div class="k">perplexity</div><div class="v" id="ppl">—</div></div>
  <div><div class="k">throughput</div><div class="v" id="tps">—</div></div>
  <div><div class="k">eta</div><div class="v" id="eta">—</div></div>
  <div><div class="k">learning rate</div><div class="v" id="lr">—</div></div>
  <div><div class="k">tokens seen</div><div class="v" id="toks">—</div></div>
</div>

<div class="card">
  <div class="k">loss history</div>
  <svg id="spark" viewBox="0 0 600 90" preserveAspectRatio="none"></svg>
</div>

<div class="card"><div class="k">loss guard</div><div class="sub" id="guard" style="margin:6px 0 0"></div></div>
<footer id="foot"></footer>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=n=>n==null?"—":n.toLocaleString();
function spark(h){
  const s=$("spark");
  if(!h||h.length<2){s.innerHTML="";return;}
  const lo=Math.min(...h),hi=Math.max(...h),span=(hi-lo)||1;
  const pts=h.map((v,i)=>[i*600/(h.length-1),85-((v-lo)/span)*78]);
  const d=pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  s.innerHTML=`<path d="${d}" fill="none" stroke="#58a6ff" stroke-width="2"/>`
    +`<text x="2" y="12" fill="#8b98a6" font-size="11">${hi.toFixed(3)}</text>`
    +`<text x="2" y="86" fill="#8b98a6" font-size="11">${lo.toFixed(3)}</text>`;
}
async function tick(){
  let d;
  try{ d=await (await fetch("/api/status",{cache:"no-store"})).json(); }
  catch(e){ $("sub").textContent="monitor unreachable — training may have ended"; $("dot").className="dot err"; return; }
  const st=d.stage||{};
  if(st.stage_list){
    $("stages").innerHTML=st.stage_list.map(x=>
      `<div class="st ${x.state}">${x.n}<b>${x.name}</b></div>`).join("");
    $("stagedetail").textContent=`stage ${st.stage}/${st.stages} — ${st.stage_name}`
      +(st.stage_detail?"   ·   "+st.stage_detail:"");
  }
  const dl=st.download;
  if(dl && st.stage===1){
    $("dlcard").hidden=false; $("trcard").hidden=true;
    $("dlfill").style.width=(dl.fraction*100).toFixed(1)+"%";
    $("dltotal").textContent=`${(dl.bytes/1e9).toFixed(1)} of ${(dl.target/1e9).toFixed(1)} GB `
      +`(${(dl.fraction*100).toFixed(1)}%)`+(dl.sft_bytes?`  ·  sft ${(dl.sft_bytes/1e6).toFixed(0)} MB`:"");
    $("dltable").innerHTML=dl.sources.map(x=>{
      const g=x.state==="done"?"✔":(x.state==="running"?"▸":"·");
      return `<tr><td class="n">${g} ${x.name}</td>`
        +`<td class="r">${(x.bytes/1e9).toFixed(2)} / ${(x.target/1e9).toFixed(2)} GB</td>`
        +`<td class="r"><span class="mini"><i style="width:${(x.fraction*100).toFixed(0)}%"></i></span></td></tr>`;
    }).join("");
  } else { $("dlcard").hidden=true; $("trcard").hidden=false; }
  $("title").textContent=d.run+" · "+(st.stage_name||d.phase);
  $("dot").className="dot "+(d.alive?"live":(d.finished?"idle":"err"));
  const stale=d.stale_seconds>=0?Math.round(d.stale_seconds)+"s ago":"—";
  $("sub").textContent = d.pre_training
    ? `${st.stage_name||"working"} · ${st.stage_detail||""}`
    : `${d.status}${d.alive?"":" (process not running)"} · ${fmt(d.params)} params · ${d.dtype} · updated ${stale}`;
  $("fill").style.width=(d.fraction*100).toFixed(2)+"%";
  $("prog").textContent=`step ${fmt(d.step)} / ${fmt(d.total)}  (${(d.fraction*100).toFixed(1)}%)`
    +(d.epochs?`   epoch ${d.epoch}/${d.epochs}`:"");
  $("msg").textContent=d.message||"";
  $("loss").textContent=d.loss!=null?d.loss.toFixed(4):"—";
  $("ppl").textContent=d.perplexity!=null?d.perplexity.toFixed(2):"—";
  $("tps").textContent=d.tokens_per_second?Math.round(d.tokens_per_second).toLocaleString()+" tok/s":"—";
  $("eta").textContent=d.eta_text+(d.finish_at?" · "+d.finish_at:"");
  $("lr").textContent=d.lr!=null?d.lr.toExponential(2):"—";
  $("toks").textContent=d.tokens_seen!=null?fmt(d.tokens_seen):"—";
  const g=d.loss_guard||{};
  $("guard").textContent=g.enabled==null?"—":
    `${g.spikes||0} spikes · ${g.rollbacks||0} rollbacks · ${g.skipped_updates||0} skipped`
    +(g.best_loss!=null?` · best ${g.best_loss.toFixed(4)} @ step ${fmt(g.best_step)}`:"");
  spark(d.history);
  $("foot").textContent="served "+d.served_at+(d.finished?" · training finished — this page will stop":"");
  if(d.finished) setTimeout(()=>{$("sub").textContent="training finished; monitor shutting down";},1000);
}
tick(); setInterval(tick,__POLLMS__);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor = None  # set on the server class before serving

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            payload = json.dumps(self.monitor.snapshot()).encode()
            self._send(payload, "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            page = (PAGE.replace("__RUN__", self.monitor.run_dir.name)
                        .replace("__POLLMS__", str(int(self.monitor.poll * 1000))))
            self._send(page.encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *_args):
        pass  # a request log per browser poll is noise, and costs a write


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="runs/kestrel-250m")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 serves the LAN; 127.0.0.1 this box only")
    ap.add_argument("--poll", type=float, default=5.0, help="Seconds between metrics reads")
    ap.add_argument("--data_dir", default="data/kestrel",
                    help="Corpus directory, for download-stage progress")
    ap.add_argument("--linger", type=float, default=300.0,
                    help="Seconds to keep serving after training ends, so a final "
                         "look is still possible. 0 exits immediately.")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="Seconds to wait for the run to appear before giving up. "
                         "Lets this start alongside a trainer that is still warming up.")
    args = ap.parse_args()

    # Lowest scheduling priority: this must never take a slice the trainer wants.
    try:
        os.nice(19)
    except (OSError, AttributeError):
        pass

    run_dir = Path(args.run_dir)
    monitor = Monitor(run_dir, args.poll, args.data_dir)
    Handler.monitor = monitor

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    ip = lan_ip()
    print(f"  Training monitor: http://{ip}:{args.port}   (LAN)", flush=True)
    print(f"                    http://127.0.0.1:{args.port}   (this machine)", flush=True)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 1.0},
                     daemon=True).start()

    # Shut down when training does. A terminal status alone is not enough --
    # the trainer writes "stopped" before its final checkpoint save returns, and
    # a resumed run rewrites the file to "running" moments later.
    deadline = time.time() + args.wait
    finished_at = None
    try:
        while True:
            time.sleep(min(5.0, max(1.0, args.poll)))
            snap = monitor.snapshot()
            if snap["alive"] or snap["status"] == "running":
                finished_at = None
                deadline = 0
                continue
            if snap["finished"]:
                if finished_at is None:
                    finished_at = time.time()
                    print(f"  Training finished ({snap['status']}). Monitor stops in "
                          f"{int(args.linger)}s.", flush=True)
                if time.time() - finished_at >= args.linger:
                    break
            elif deadline and time.time() > deadline:
                print("  No run found; monitor exiting.", flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print("  Monitor stopped.", flush=True)


if __name__ == "__main__":
    main()
