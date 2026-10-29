# redis runs on port 6379
# start command: docker run -d -p 6379:6379 redis:7-alpine

from __future__ import annotations

import json
import hashlib
import hmac
import time
import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Condition
from typing import Callable, Optional

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent
    redis = None

# Allow importing config from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg

log = logging.getLogger("fission.bus")


@dataclass
class MCPMessage:
    """
    A message on the FISSION MCP bus.
    Travels over Redis Streams (or an in-memory fallback in local dev).
    """
    sender: str
    recipient: str
    msg_type: str          # "tool_call_request" | "tool_call_result" | "task"
    payload: dict
    hop_count: int = 1
    timestamp: float = 0.0

    def to_redis_dict(self) -> dict:
        """Serialise for Redis XADD — payload must be a string."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "msg_type": self.msg_type,
            "payload": json.dumps(self.payload),
            "hop_count": str(self.hop_count),
            "timestamp": str(self.timestamp),
        }

    @classmethod
    def from_redis_dict(cls, data: dict) -> "MCPMessage":
        """Deserialise from Redis XREAD response."""
        d = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()
        }
        return cls(
            sender=d["sender"],
            recipient=d["recipient"],
            msg_type=d["msg_type"],
            payload=json.loads(d["payload"]),
            hop_count=int(d.get("hop_count", 1)),
            timestamp=float(d.get("timestamp", 0.0)),
        )


class _InMemoryBusState:
    """Shared process-local bus state used when Redis is unavailable."""
    streams: dict[str, deque[MCPMessage]] = defaultdict(deque)
    conditions: dict[str, Condition] = defaultdict(Condition)
    agent_set: set[str] = set()
    log: list[dict] = []
    context: list[dict] = []


# ══════════════════════════════════════════════════════════════
#  Redis / in-memory MCP Bus
# ══════════════════════════════════════════════════════════════

class RedisMCPBus:
    """
    Redis Streams-backed MCP message bus.

    Local/dev convenience:
        If the redis package is missing, or FISSION_BUS_BACKEND=memory,
        the bus transparently falls back to an in-process memory backend.
        This keeps run.py --no-vlm usable without extra infrastructure.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, **_ignored):
        """
        Args:
            host / port: optional Redis overrides (orchestrator/executor pass these).
                         Defaults come from cfg.mcp.redis_host / redis_port.
            **_ignored: accepts extra kwargs so older call sites don't crash.
        """
        self._secret = cfg.defense.hmac_secret.encode()
        self._pending: dict[str, deque[MCPMessage]] = defaultdict(deque)
        self._host = host if host is not None else cfg.mcp.redis_host
        self._port = int(port if port is not None else cfg.mcp.redis_port)

        backend_override = os.environ.get("FISSION_BUS_BACKEND", "").strip().lower()
        self._backend = "redis"
        self._r = None

        if backend_override == "memory" or redis is None:
            self._backend = "memory"
            reason = "FISSION_BUS_BACKEND=memory" if backend_override == "memory" else "redis package not installed"
            log.warning(f"Using in-memory MCP bus ({reason})")
        else:
            self._r = redis.Redis(
                host=self._host,
                port=self._port,
                decode_responses=False,
            )
            self._verify_connection()
            log.info(
                f"MCP bus initialised — Redis at "
                f"{self._host}:{self._port}"
            )

    def _verify_connection(self):
        if self._r is None:
            return
        try:
            self._r.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Redis at "
                f"{self._host}:{self._port}\n"
                f"Start Redis locally: docker run -d -p 6379:6379 redis:7-alpine\n"
                f"Or use memory backend: FISSION_BUS_BACKEND=memory\n"
                f"Original error: {e}"
            )

    # ── Stream key helpers ───────────────────────────────────────

    def _stream_key(self, agent_name: str) -> bytes:
        return f"{cfg.mcp.stream_prefix}{agent_name}".encode()

    def _cursor_key(self, agent_name: str) -> bytes:
        return f"fission:cursor:{agent_name}".encode()

    # ── HMAC signing ─────────────────────────────────────────────

    def sign(
        self,
        tool_name: str,
        arguments: dict,
        intent_text: str,
        task_id: str,
    ) -> str:
        payload = json.dumps({
            "tool": tool_name,
            "args": arguments,
            "intent": intent_text,
            "task_id": task_id,
        }, sort_keys=True).encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def verify(
        self,
        tool_name: str,
        arguments: dict,
        intent_text: str,
        task_id: str,
        provided_sig: str,
    ) -> bool:
        expected = self.sign(tool_name, arguments, intent_text, task_id)
        return hmac.compare_digest(provided_sig, expected)

    # ── Agent registration ────────────────────────────────────────

    def register(self, agent_name: str):
        """Register an agent on the bus."""
        if self._backend == "memory":
            _InMemoryBusState.agent_set.add(agent_name)
            _InMemoryBusState.streams[agent_name]
            _InMemoryBusState.conditions[agent_name]
            log.info(f"Agent registered on in-memory bus: {agent_name}")
            return

        assert self._r is not None
        self._r.sadd(cfg.mcp.agent_set_key, agent_name)
        stream_key = self._stream_key(agent_name)
        try:
            self._r.xinfo_stream(stream_key)
        except redis.ResponseError:
            self._r.xadd(stream_key, {"_init": "1"})
        log.info(f"Agent registered on bus: {agent_name}")

    # ── Send ─────────────────────────────────────────────────────

    def send(self, msg: MCPMessage, timeout: Optional[float] = None):
        """Put a message on the recipient's stream/queue."""
        msg.timestamp = time.time()
        log_entry = {
            "sender": msg.sender,
            "recipient": msg.recipient,
            "msg_type": msg.msg_type,
            "payload": msg.payload,
            "hop_count": msg.hop_count,
            "timestamp": msg.timestamp,
        }

        if self._backend == "memory":
            cond = _InMemoryBusState.conditions[msg.recipient]
            with cond:
                _InMemoryBusState.streams[msg.recipient].append(msg)
                _InMemoryBusState.log.append(log_entry)
                cond.notify_all()
        else:
            assert self._r is not None
            stream_key = self._stream_key(msg.recipient)
            self._r.xadd(stream_key, msg.to_redis_dict())
            self._r.rpush(cfg.mcp.log_key, json.dumps(log_entry))

        log.info(
            f"[BUS] {msg.sender} → {msg.recipient} "
            f"[{msg.msg_type}] hop={msg.hop_count}"
        )

    # ── Receive ──────────────────────────────────────────────────

    def _pending_pop(self, agent_name: str, predicate: Callable[[MCPMessage], bool] | None) -> Optional[MCPMessage]:
        queue = self._pending[agent_name]
        if not queue:
            return None

        if predicate is None:
            return queue.popleft()

        for _ in range(len(queue)):
            msg = queue.popleft()
            if predicate(msg):
                return msg
            queue.append(msg)
        return None

    def receive(self, agent_name: str, timeout: float = 30.0) -> Optional[MCPMessage]:
        """Receive the next available message for an agent."""
        return self.receive_matching(agent_name, predicate=None, timeout=timeout)

    def receive_matching(
        self,
        agent_name: str,
        predicate: Callable[[MCPMessage], bool] | None,
        timeout: float = 30.0,
    ) -> Optional[MCPMessage]:
        """
        Receive the next message matching a predicate.

        Non-matching messages are buffered in-memory so they can be
        retrieved by later calls instead of being silently dropped.
        """
        pending = self._pending_pop(agent_name, predicate)
        if pending is not None:
            return pending

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._backend == "memory":
                remaining = max(0.0, deadline - time.time())
                cond = _InMemoryBusState.conditions[agent_name]
                with cond:
                    if not _InMemoryBusState.streams[agent_name]:
                        cond.wait(timeout=min(remaining, 0.5))
                    if _InMemoryBusState.streams[agent_name]:
                        msg = _InMemoryBusState.streams[agent_name].popleft()
                    else:
                        msg = None
            else:
                msg = self._receive_from_redis(agent_name, deadline)

            if msg is None:
                continue

            if predicate is None or predicate(msg):
                return msg

            self._pending[agent_name].append(msg)

        log.warning(f"Timeout waiting for message on '{agent_name}' stream")
        return None

    def _receive_from_redis(self, agent_name: str, deadline: float) -> Optional[MCPMessage]:
        assert self._r is not None
        stream_key = self._stream_key(agent_name)
        cursor_key = self._cursor_key(agent_name)
        last_id = self._r.get(cursor_key) or b"0"

        remaining_ms = max(1, int((deadline - time.time()) * 1000))
        results = self._r.xread(
            {stream_key: last_id},
            count=1,
            block=min(remaining_ms, 1000),
        )
        if not results:
            return None

        _, entries = results[0]
        for entry_id, data in entries:
            last_id = entry_id
            self._r.set(cursor_key, last_id)
            d = {
                k.decode() if isinstance(k, bytes) else k: v
                for k, v in data.items()
            }
            if "_init" in d:
                continue
            return MCPMessage.from_redis_dict(data)
        return None

    # ── Context buffer (R5b target) ───────────────────────────────

    def write_context(self, agent_name: str, content: str):
        payload = {
            "agent": agent_name,
            "content": content,
            "timestamp": time.time(),
        }
        if self._backend == "memory":
            _InMemoryBusState.context.append(payload)
        else:
            assert self._r is not None
            self._r.rpush(cfg.mcp.context_buffer_key, json.dumps(payload))

    def read_context(self) -> list[dict]:
        if self._backend == "memory":
            return list(_InMemoryBusState.context)
        assert self._r is not None
        raw = self._r.lrange(cfg.mcp.context_buffer_key, 0, -1)
        return [json.loads(r) for r in raw]

    # ── Log access ────────────────────────────────────────────────

    def get_log(self) -> list[dict]:
        if self._backend == "memory":
            return list(_InMemoryBusState.log)
        assert self._r is not None
        raw = self._r.lrange(cfg.mcp.log_key, 0, -1)
        return [json.loads(r) for r in raw]

    def dump_log(self):
        print("\n── MCP Bus Log ──")
        for entry in self.get_log():
            if entry.get("msg_type") not in ("_init",):
                print(
                    f"  {entry['sender']} → {entry['recipient']} "
                    f"[{entry['msg_type']}] hop={entry['hop_count']}"
                )

    def flush(self):
        """Clear all FISSION bus state. Use between test runs."""
        self._pending.clear()
        if self._backend == "memory":
            _InMemoryBusState.streams.clear()
            _InMemoryBusState.conditions.clear()
            _InMemoryBusState.agent_set.clear()
            _InMemoryBusState.log.clear()
            _InMemoryBusState.context.clear()
            log.info("In-memory bus flushed — all FISSION state cleared")
            return

        assert self._r is not None
        for key in self._r.scan_iter("fission:*"):
            self._r.delete(key)
        log.info("Bus flushed — all FISSION state cleared")