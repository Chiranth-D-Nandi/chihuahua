"""
FISSION — orchestrator/hmac_signer.py
=======================================
HMAC-SHA256 intent binding.
Signs every orchestrator tool call so B4 TCIV can verify
that the call was produced for THIS specific intent and was
not tampered with in transit.

ABHYUDAYA — orchestrator side.
"""

import hashlib
import hmac
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg


class HMACSigner:

    def __init__(self):
        self._secret = cfg.defense.hmac_secret.encode()

    def sign(
        self,
        tool_name: str,
        arguments: dict,
        intent_text: str,
        task_id: str,
    ) -> str:
        """
        Produce HMAC-SHA256 over the canonical tool call payload.

        IMPORTANT:
            This payload contract must stay byte-for-byte aligned with
            defenses.b4_tciv.tciv.TCIV.sign().
        """
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

