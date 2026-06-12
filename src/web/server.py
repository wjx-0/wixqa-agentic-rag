from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.agentic.runtime_factory import (
    DialogueRuntimeConfig,
    DialogueRuntime,
    build_demo_dialogue_runtime,
    build_real_dialogue_runtime,
)
from src.utils.io_utils import dumps_json, model_to_dict
from src.utils.text_utils import compact_text


STATIC_DIR = Path(__file__).resolve().parent / "static"


class DialogueWebServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        runtime: DialogueRuntime,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.runtime = runtime


class DialogueRequestHandler(BaseHTTPRequestHandler):
    server: DialogueWebServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/"}:
            self.serve_static_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/static/app.css":
            self.serve_static_file(STATIC_DIR / "app.css", "text/css; charset=utf-8")
            return
        if path == "/static/app.js":
            self.serve_static_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if path == "/api/health":
            self.send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat/stream":
            self.handle_chat_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_chat_stream(self) -> None:
        try:
            payload = self.read_json_body()
            conversation_id = compact_text(payload.get("conversation_id")) or "web_conversation"
            message = compact_text(payload.get("message"))
            profile = compact_text(payload.get("profile")) or "fast"
            if not message:
                self.send_json({"error": "message must not be empty"}, status=HTTPStatus.BAD_REQUEST)
                return
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for event in self.server.runtime.orchestrator.run_turn(
                conversation_id,
                message,
                profile=profile,
            ):
                self.write_ndjson({"type": "event", "event": model_to_dict(event)})
        except Exception as exc:  # pragma: no cover - keeps browser stream readable.
            self.write_ndjson({"type": "error", "error": str(exc)})

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def write_ndjson(self, value: dict[str, Any]) -> None:
        self.wfile.write(dumps_json(value, indent=False) + b"\n")
        self.wfile.flush()

    def send_json(self, value: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = dumps_json(value, indent=False)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_server(
    *,
    host: str,
    port: int,
    mode: str,
    config: DialogueRuntimeConfig | None = None,
) -> DialogueWebServer:
    runtime = (
        build_demo_dialogue_runtime()
        if mode == "demo"
        else build_real_dialogue_runtime(config)
    )
    try:
        return DialogueWebServer((host, port), DialogueRequestHandler, runtime=runtime)
    except Exception:
        runtime.close()
        raise
