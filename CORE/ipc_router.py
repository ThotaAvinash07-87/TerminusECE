"""Inter-Process Communication (IPC) daemon and client for cross-terminal data exchange."""

from __future__ import annotations
import asyncio
import json
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set


class IPCRouter:
    """Local IPC Socket Server that synchronizes data, signals, and events between terminal sessions."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.shared_memory: Dict[str, Any] = {}
        self.subscribers: Dict[str, Set[asyncio.StreamWriter]] = {}
        self.server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handles single connected client socket."""
        subscribed_topics: Set[str] = set()

        try:
            while self._running:
                line = await reader.readline()
                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue

                msg_type = msg.get("type")
                topic = msg.get("topic", "")
                payload = msg.get("payload", {})
                req_id = msg.get("id")

                response: Dict[str, Any] = {"id": req_id, "status": "ok"}

                if msg_type == "set_var":
                    var_name = payload.get("name")
                    var_val = payload.get("value")
                    if var_name:
                        self.shared_memory[var_name] = var_val
                        response["result"] = "saved"
                        # Notify subscribers of this variable
                        await self._broadcast(f"var:{var_name}", {"name": var_name, "value": var_val})

                elif msg_type == "get_var":
                    var_name = payload.get("name")
                    response["value"] = self.shared_memory.get(var_name)

                elif msg_type == "list_vars":
                    response["variables"] = list(self.shared_memory.keys())

                elif msg_type == "subscribe":
                    if topic:
                        if topic not in self.subscribers:
                            self.subscribers[topic] = set()
                        self.subscribers[topic].add(writer)
                        subscribed_topics.add(topic)
                        response["subscribed"] = topic

                elif msg_type == "publish":
                    if topic:
                        await self._broadcast(topic, payload)
                        response["published"] = topic

                elif msg_type == "ping":
                    response["pong"] = time.time()

                else:
                    response["status"] = "unknown_command"

                out_data = (json.dumps(response) + "\n").encode("utf-8")
                writer.write(out_data)
                await writer.drain()

        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            # Cleanup subscriptions
            for top in subscribed_topics:
                if top in self.subscribers and writer in self.subscribers[top]:
                    self.subscribers[top].remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _broadcast(self, topic: str, payload: Any) -> None:
        """Sends broadcast to all subscribers of a topic."""
        writers = list(self.subscribers.get(topic, set()))
        if not writers:
            return
        
        msg = json.dumps({"type": "event", "topic": topic, "payload": payload}) + "\n"
        data = msg.encode("utf-8")

        for w in writers:
            try:
                w.write(data)
                await w.drain()
            except Exception:
                pass

    async def _start_server(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self._running = True
        async with self.server:
            await self.server.serve_forever()

    def start_background(self) -> None:
        """Starts the IPC server daemon in a background thread."""
        if self._running:
            return

        def run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._start_server())
            except Exception:
                pass

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.05)  # brief wait for bind

    def stop(self) -> None:
        """Stops the IPC server."""
        self._running = False
        if self.server:
            self.server.close()
        if self._loop:
            self._loop.stop()


class IPCClient:
    """Synchronous / Asynchronous helper for sending messages to the IPC router."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def is_server_running(self) -> bool:
        """Checks if IPC server is listening."""
        try:
            with socket.create_connection((self.host, self.port), timeout=0.2):
                return True
        except (socket.error, socket.timeout):
            return False

    def send_command(self, msg_type: str, payload: Optional[Dict[str, Any]] = None, topic: str = "") -> Optional[Dict[str, Any]]:
        """Sends command synchronously to the IPC server and returns response."""
        payload = payload or {}
        req = {
            "type": msg_type,
            "topic": topic,
            "payload": payload,
            "id": f"req_{int(time.time() * 1000)}"
        }

        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                msg_bytes = (json.dumps(req) + "\n").encode("utf-8")
                sock.sendall(msg_bytes)
                
                sock.settimeout(self.timeout)
                file_obj = sock.makefile("r", encoding="utf-8")
                line = file_obj.readline()
                if line:
                    return json.loads(line.strip())
        except Exception:
            return None
        return None

    def set_variable(self, name: str, value: Any) -> bool:
        """Stores a variable in the shared IPC memory."""
        res = self.send_command("set_var", {"name": name, "value": value})
        return bool(res and res.get("status") == "ok")

    def get_variable(self, name: str) -> Any:
        """Retrieves a variable from the shared IPC memory."""
        res = self.send_command("get_var", {"name": name})
        if res and res.get("status") == "ok":
            return res.get("value")
        return None

    def publish_signal(self, topic: str, data: Any) -> bool:
        """Publishes an event or signal trace to a topic."""
        res = self.send_command("publish", data if isinstance(data, dict) else {"data": data}, topic=topic)
        return bool(res and res.get("status") == "ok")
