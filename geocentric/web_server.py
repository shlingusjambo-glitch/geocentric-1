"""Bundled LAN inference server. No Node, cloud service, or model uploads required."""
from __future__ import annotations

import ipaddress
import json
import math
import socket
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import torch

from geocentric.chat import DEFAULT_SYSTEM
from geocentric.checkpoint import load_model_and_tokenizer
from geocentric.device import resolve_dtype, select_device
from geocentric.generate import build_chat_prompt, stream_text

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
    def __init__(self, model, tokenizer, stage="pretrained", dtype=torch.float32, defaults=None):
        self.model, self.tokenizer, self.stage, self.dtype = model, tokenizer, stage, dtype
        self.defaults = defaults or {}
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
        return prompt, options, request_id

    def stream(self, prompt, options, event, stats):
        device = next(self.model.parameters()).device
        # Autocast/inference_mode are thread-local: enter in the request thread.
        with torch.inference_mode(), torch.autocast(device.type, dtype=self.dtype,
                                                    enabled=self.dtype != torch.float32):
            yield from stream_text(self.model, self.tokenizer, prompt, cancel_event=event,
                                   stats=stats, loop_guard=True, **options)


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, engine):
        self.engine = engine
        super().__init__(address, ChatHandler)


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "Geocentric"

    def log_message(self, format, *args):
        # Chat text is never logged by the server.
        pass

    def send_response_headers(self, status, content_type, preview=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        policy = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; "
                  "style-src 'unsafe-inline'; img-src data:; font-src data:; "
                  "connect-src 'none'; frame-src 'none'; object-src 'none'; "
                  "form-action 'none'; base-uri 'none'; frame-ancestors 'self'") if preview else (
                  "default-src 'self'; script-src 'self'; style-src 'self'; "
                  "img-src 'self' data:; connect-src 'self'; frame-src 'self'; "
                  "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.send_header("Content-Security-Policy", policy)
        self.end_headers()

    def json_response(self, status, payload):
        self.send_response_headers(status, "application/json")
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/model":
            return self.json_response(200, self.server.engine.info())
        if path == "/health":
            return self.json_response(200, {"status": "ok"})
        if path not in ASSETS:
            return self.json_response(404, {"error": "Not found"})
        filename, content_type = ASSETS[path]
        self.send_response_headers(200, content_type)
        self.wfile.write((STATIC / filename).read_bytes())

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
            if self.path == "/api/preview":
                if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                    return self.json_response(415, {"error": "Expected form-encoded source"})
                fields = parse_qs(body.decode("utf-8"), max_num_fields=2, keep_blank_values=True)
                codes = fields.get("code", [])
                if len(codes) != 1 or len(codes[0].encode("utf-8")) > 60000:
                    return self.json_response(400, {"error": "Send one code field, up to 60 KB"})
                self.send_response_headers(200, "text/html", preview=True)
                self.wfile.write(codes[0].encode("utf-8"))
                return
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
        if self.path != "/api/chat":
            return self.json_response(404, {"error": "Not found"})
        try:
            prompt, options, request_id = engine.validate(payload)
        except ValueError as exc:
            return self.json_response(400, {"error": str(exc)})
        if not engine.lock.acquire(blocking=False):
            return self.json_response(429, {"error": "The model is answering another request. Try again shortly."})
        event, stats = threading.Event(), {}
        with engine.request_lock:
            engine.requests[request_id] = event
        try:
            self.send_response_headers(200, "application/x-ndjson")
            self.emit({"type": "start", "request_id": request_id})
            for chunk in engine.stream(prompt, options, event, stats):
                self.emit({"type": "delta", "text": chunk})
            self.emit({"type": "done", **stats})
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

    def emit(self, payload):
        self.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
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
    engine = ChatEngine(model, tokenizer, stage, dtype, defaults)
    with ChatServer((args.host, args.port), engine) as server:
        local, lan = network_urls(args.host, server.server_port)
        print(f"\n{model.config.model_name} · {stage} · {device} · {dtype}", flush=True)
        print(f"  Local: {local}", flush=True)
        for url in lan:
            print(f"  LAN / Wi-Fi: {url}", flush=True)
        if not lan and args.host == "0.0.0.0":
            print("  No LAN address detected; check your network connection.", flush=True)
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
