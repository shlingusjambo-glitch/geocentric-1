"""Bundled LAN inference server. No Node, cloud service, or model uploads required."""
from __future__ import annotations

import gzip
import ipaddress
import json
import math
import socket
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch

from geocentric.chat import DEFAULT_SYSTEM
from geocentric.checkpoint import load_model_and_tokenizer
from geocentric.device import resolve_dtype, select_device
from geocentric.generate import build_chat_prompt, stream_text
from geocentric.load import LoadMonitor, paced
from geocentric.testers import TesterRegistry
from geocentric.transcripts import TranscriptStore

STATIC = Path(__file__).with_name("web")
ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css"),
          "/mascot.png": ("mascot.png", "image/png"), "/logo.png": ("logo.png", "image/png")}


def network_urls(host, port):
    """Return localhost and IPv4 interface URLs without changing firewall/router state."""
    addresses = set()
    if host in {"0.0.0.0", ""}:
        try:
            addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        # UDP connect selects a route without sending an application packet.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("192.0.2.1", 80))
                addresses.add(sock.getsockname()[0])
        except OSError:
            pass
        # Enumerate all IPv4 interfaces, including Wi-Fi alongside Ethernet/VPN.
        try:
            import fcntl
            import struct
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                for _, name in socket.if_nameindex():
                    request = struct.pack("256s", name[:15].encode())
                    import sys
                    code = 0x8915 if sys.platform.startswith("linux") else 0xc0206921
                    try:
                        result = fcntl.ioctl(sock.fileno(), code, request)
                        addresses.add(socket.inet_ntoa(result[20:24]))
                    except OSError:
                        continue
        except (ImportError, OSError):
            pass
    elif host not in {"127.0.0.1", "localhost"}:
        addresses.add(socket.gethostbyname(host))
    lan = sorted(a for a in addresses if a not in {"0.0.0.0", "127.0.0.1"}
                 and not ipaddress.ip_address(a).is_loopback)
    local_host = "127.0.0.1" if host in {"", "0.0.0.0", "localhost"} else host
    local = f"http://{local_host}:{port}"
    return local, [f"http://{a}:{port}" for a in lan]


class ChatEngine:
    def __init__(self, model, tokenizer, stage="pretrained", dtype=torch.float32, defaults=None,
                 transcripts=None, testers=None):
        self.model, self.tokenizer, self.stage, self.dtype = model, tokenizer, stage, dtype
        self.defaults = defaults or {}
        self.transcripts = transcripts
        self.testers = testers
        self.load = LoadMonitor()
        self.lock = threading.Lock()
        self.requests = {}
        self.request_lock = threading.Lock()

    def info(self):
        return dict(name=self.model.config.model_name, parameters=self.model.num_params(),
                    context=self.model.config.block_size, stage=self.stage,
                    mode="chat" if self.stage in {"sft", "vision"} else "base",
                    device=str(next(self.model.parameters()).device), dtype=str(self.dtype),
                    defaults=self.defaults, vision=bool(self.model.config.vision))

    def validate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
            raise ValueError("Send between 1 and 200 messages")
        for row in messages:
            if not isinstance(row, dict) or row.get("role") not in ("user", "assistant"):
                raise ValueError("Messages must have user or assistant roles")
            if not isinstance(row.get("content"), str) or len(row["content"]) > 100000:
                raise ValueError("Message content must be text, up to 100,000 characters")
        if messages[-1]["role"] != "user" or not messages[-1]["content"].strip():
            raise ValueError("The last message must be a nonempty user message")
        mode = payload.get("mode", self.defaults.get("mode", "auto"))
        if mode not in ("auto", "chat", "base"):
            raise ValueError("Unknown conversation mode")
        if mode == "auto":
            mode = self.info()["mode"]
        system = payload.get("system", self.defaults.get("system") or DEFAULT_SYSTEM)
        if not isinstance(system, str) or len(system) > 16000:
            raise ValueError("System prompt must be text, up to 16,000 characters")
        prompt = (build_chat_prompt(messages, system) if mode == "chat"
                  else "\n\n".join(row["content"] for row in messages))
        ranges = {"max_new_tokens": (1, min(4096, self.model.config.block_size - 1), int),
                  "temperature": (0, 2, float), "top_k": (0, self.model.config.vocab_size, int),
                  "top_p": (0, 1, float), "min_p": (0, 1, float),
                  "repetition_penalty": (1, 2, float)}
        options = {}
        fallback = dict(max_new_tokens=256, temperature=.8, top_k=50, top_p=.95,
                        min_p=.05, repetition_penalty=1.25)
        for key, (low, high, kind) in ranges.items():
            default = min(high, self.defaults.get(key, fallback[key]))
            value = payload.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {key}")
            if not low <= value <= high or (kind is int and int(value) != value):
                raise ValueError(f"{key} must be between {low} and {high}")
            options[key] = kind(value)
        request_id = payload.get("request_id") or str(uuid.uuid4())
        try:
            uuid.UUID(request_id)
        except (ValueError, TypeError, AttributeError):
            raise ValueError("request_id must be a UUID")
        training_consent = payload.get("training_consent", False)
        if not isinstance(training_consent, bool):
            raise ValueError("training_consent must be true or false")
        tester = payload.get("tester_key")
        if tester is not None and not isinstance(tester, str):
            raise ValueError("tester_key must be text")
        return prompt, options, request_id, training_consent, tester

    def stream(self, prompt, options, event, stats, tokens_per_second=None):
        """Generate, held to the serving rate cap so one caller cannot take the
        whole machine. The cap is read once per request, not per token."""
        device = next(self.model.parameters()).device
        # Autocast/inference_mode are thread-local: enter in the request thread.
        with torch.inference_mode(), torch.autocast(device.type, dtype=self.dtype,
                                                    enabled=self.dtype != torch.float32):
            chunks = stream_text(self.model, self.tokenizer, prompt, cancel_event=event,
                                 stats=stats, loop_guard=True, **options)
            yield from paced(chunks, tokens_per_second)


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, engine):
        self.engine = engine
        super().__init__(address, ChatHandler)


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "Geocentric"

    def log_message(self, format, *args):
        # No access log. Chat text is retained only by TranscriptStore, and only
        # when the operator started the server with --transcripts.
        pass

    def wants_gzip(self):
        return "gzip" in self.headers.get("Accept-Encoding", "").lower()

    def send_response_headers(self, status, content_type, gzipped=False, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        if gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy",
                         "geolocation=(), camera=(), microphone=(), payment=(), usb=()")
        # Only meaningful behind TLS; harmless on a LAN server, which browsers
        # ignore it on because the connection is not HTTPS.
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        policy = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                  "img-src 'self' data:; connect-src 'self'; frame-src 'none'; "
                  "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                  "form-action 'self'")
        self.send_header("Content-Security-Policy", policy)
        self.end_headers()

    def json_response(self, status, payload):
        # Compact separators: no spaces to pay for on a phone over Wi-Fi.
        body = json.dumps(payload, separators=(",", ":")).encode()
        gzipped = self.wants_gzip() and len(body) > 512
        if gzipped:
            body = gzip.compress(body, 6)
        self.send_response_headers(status, "application/json", gzipped, len(body))
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/model":
            return self.json_response(200, self.server.engine.info())
        if path == "/api/status":
            return self.json_response(200, self.server.engine.load.state())
        if path == "/health":
            return self.json_response(200, {"status": "ok"})
        if path not in ASSETS:
            return self.json_response(404, {"error": "Not found"})
        filename, content_type = ASSETS[path]
        body = self._asset(filename)
        # PNGs are already compressed; gzip would only add bytes and CPU.
        gzipped = self.wants_gzip() and not filename.endswith(".png")
        if gzipped:
            body = gzip.compress(body, 6)
        self.send_response_headers(200, content_type, gzipped, len(body))
        self.wfile.write(body)

    _asset_cache = {}

    @classmethod
    def _asset(cls, filename):
        """Read once. These files do not change while the server is running."""
        if filename not in cls._asset_cache:
            cls._asset_cache[filename] = (STATIC / filename).read_bytes()
        return cls._asset_cache[filename]

    def do_POST(self):
        # Require same-origin browser writes. No permissive CORS, proxy URLs,
        # shell endpoints, arbitrary file access, or checkpoint upload endpoints.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.json_response(403, {"error": "Cross-origin requests are not allowed"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 256 * 1024:
                return self.json_response(413, {"error": "Request exceeds 256 KiB"})
            self.connection.settimeout(120)
            body = self.rfile.read(length)
            payload = json.loads(body)
        except (ValueError, OSError):
            return self.json_response(400, {"error": "Invalid JSON request"})
        engine = self.server.engine
        if self.path == "/api/cancel":
            if not isinstance(payload, dict) or not isinstance(payload.get("request_id"), str):
                return self.json_response(400, {"error": "Expected a JSON object"})
            with engine.request_lock:
                event = engine.requests.get(payload.get("request_id"))
                if event:
                    event.set()
            return self.json_response(200, {"cancelled": event is not None})
        if self.path == "/api/tester":
            if not isinstance(payload, dict):
                return self.json_response(400, {"error": "Expected a JSON object"})
            if not engine.testers:
                return self.json_response(404, {"error": "Tester access is not enabled"})
            label = engine.testers.verify(payload.get("key"))
            if not label:
                return self.json_response(403, {"error": "That access key was not recognised"})
            return self.json_response(200, {"ok": True, "label": label})
        if self.path == "/api/report":
            return self.handle_report(payload, engine)
        if self.path != "/api/chat":
            return self.json_response(404, {"error": "Not found"})
        try:
            prompt, options, request_id, training_consent, tester_key = engine.validate(payload)
        except ValueError as exc:
            return self.json_response(400, {"error": str(exc)})
        if not engine.lock.acquire(blocking=False):
            engine.load.note_contention()
            return self.json_response(429, {"error": "The model is answering another request. Try again shortly."})
        event, stats = threading.Event(), {}
        with engine.request_lock:
            engine.requests[request_id] = event
        try:
            load = engine.load.state()
            self.send_response_headers(200, "application/x-ndjson")
            self.emit({"type": "start", "request_id": request_id, "load": load})
            reply = []
            for chunk in engine.stream(prompt, options, event, stats,
                                       load["tokens_per_second"]):
                reply.append(chunk)
                self.emit({"type": "delta", "text": chunk})
            self.emit({"type": "done", **stats})
            tester = engine.testers.verify(tester_key) if (engine.testers and tester_key) else None
            # Nothing is retained unless the user opted in, and an authorised
            # tester may be under 16, so their exchanges are never retained for
            # training at all.
            if engine.transcripts and training_consent and not tester:
                engine.transcripts.record(prompt, "".join(reply), True,
                                          request_id, stats)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            event.set()
        except Exception as exc:
            event.set()
            try:
                self.emit({"type": "error", "error": "Generation failed: " + type(exc).__name__})
            except OSError:
                pass
        finally:
            with engine.request_lock:
                engine.requests.pop(request_id, None)
            engine.lock.release()

    def handle_report(self, payload, engine):
        """A report the user explicitly chose to send, having seen its contents."""
        if not isinstance(payload, dict):
            return self.json_response(400, {"error": "Expected a JSON object"})
        wrong = (payload.get("what_went_wrong") or "").strip()
        if not wrong:
            return self.json_response(400, {"error": "Tell us what went wrong"})
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) > 200:
            return self.json_response(400, {"error": "Send up to 200 messages"})
        if engine.transcripts is None:
            # Nowhere to put it; say so rather than pretend it was received.
            return self.json_response(503, {"error": "Reporting is not enabled on this server"})
        report = {
            "what_went_wrong": wrong[:4000],
            "why": (payload.get("why") or "").strip()[:4000],
            "expected": (payload.get("expected") or "").strip()[:4000],
            "notes": (payload.get("notes") or "").strip()[:4000],
            "messages": [
                {"role": str(m.get("role", ""))[:16], "content": str(m.get("content", ""))[:100000]}
                for m in messages if isinstance(m, dict)
            ],
        }
        if not engine.transcripts.record_report(report):
            return self.json_response(500, {"error": "Could not store the report"})
        return self.json_response(200, {"ok": True})

    def emit(self, payload):
        # Compact separators only. The stream is not gzipped on purpose: a
        # compressor buffers, and buffering is what makes a token stream feel
        # slow. Each frame is a few bytes of JSON around the text itself.
        self.wfile.write(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )
        self.wfile.flush()


def serve(args):
    if args.image:
        raise ValueError("Use try --terminal --image for image input; the web client currently accepts text")
    device = select_device()
    dtype = resolve_dtype(device, args.dtype)
    if device.type == "cpu" and args.dtype == "auto":
        dtype = torch.float32
    model, tokenizer, stage = load_model_and_tokenizer(args.model_dir, device=device,
                                                      checkpoint_name=args.checkpoint, with_stage=True)
    defaults = {key: getattr(args, key) for key in ("max_new_tokens", "temperature", "top_k",
                "top_p", "min_p", "repetition_penalty", "system", "mode")}
    transcripts = (TranscriptStore(args.transcripts, getattr(args, "transcript_days", 30))
                   if getattr(args, "transcripts", None) else None)
    testers = TesterRegistry(args.tester_keys) if getattr(args, "tester_keys", None) else None
    engine = ChatEngine(model, tokenizer, stage, dtype, defaults, transcripts, testers)
    with ChatServer((args.host, args.port), engine) as server:
        local, lan = network_urls(args.host, server.server_port)
        print(f"\n{model.config.model_name} · {stage} · {device} · {dtype}", flush=True)
        print(f"  Local: {local}", flush=True)
        for url in lan:
            print(f"  LAN / Wi-Fi: {url}", flush=True)
        if not lan and args.host == "0.0.0.0":
            print("  No LAN address detected; check your network connection.", flush=True)
        if testers:
            print(f"  Authorised-tester access enabled: {len(testers.records)} key(s).", flush=True)
        if transcripts:
            print(f"  Opt-in retention on: consented conversations and reports go to "
                  f"{transcripts.dir} and expire after {transcripts.retention_days} days.",
                  flush=True)
        else:
            print("  Retention off: nothing is stored, and reporting is disabled.", flush=True)
        print(f"  Serving at up to {engine.load.normal:.0f} tokens/sec per request, "
              f"{engine.load.high:.0f} under load.", flush=True)
        print("  Chats stay in this browser. Anyone on the reachable LAN can use this server.\n"
              "  Press Ctrl+C to stop. Use --host 127.0.0.1 for localhost only.\n", flush=True)
        if not args.no_browser:
            threading.Timer(.3, lambda: webbrowser.open(local)).start()
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            with engine.request_lock:
                for event in engine.requests.values():
                    event.set()
            print("\nChat server stopped.")
