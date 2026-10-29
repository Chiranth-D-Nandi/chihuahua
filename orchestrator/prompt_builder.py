import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg


class PromptBuilder:

    def build_system_prompt(self) -> str:
        tools_schema = {}
        for name, tool in cfg.mcp.registered_tools.items():
            if cfg.mcp.role_executor not in tool["allowed_roles"]:
                continue
            # config stores types under "schema" as {arg_name: type}
            # and semantic descriptions under "arg_semantics" as {arg_name: str}
            schema = tool.get("schema", {})
            semantics = tool.get("arg_semantics", {})
            tools_schema[name] = {
                "description": tool["description"],
                "arguments": {
                    arg_name: {
                        "type": arg_type.__name__ if isinstance(arg_type, type) else str(arg_type),
                        "description": semantics.get(arg_name, ""),
                    }
                    for arg_name, arg_type in schema.items()
                },
            }

        tool_descriptions = json.dumps(tools_schema, indent=2)

        exact_names: list[str] = []
        for tool in tools_schema.values():
            exact_names.extend(tool["arguments"].keys())
        no_alias_line = (
            "- Use EXACTLY these argument names: "
            + ", ".join(exact_names)
            + ". Do not invent aliases."
        ) if exact_names else "- Use argument names exactly as listed. Do not invent aliases."

        return f"""You are a task executor agent in a secure multi-agent pipeline.
Given a task, respond with ONE JSON tool call only.

Format:
{{
  "tool": "<tool_name>",
  "arguments": {{
    "<param>": "<value>"
  }}
}}

Available tools and their EXACT argument names:
{tool_descriptions}

Rules:
- Respond with ONE JSON object only.
- Do not escape underscores.
- Use exact tool names and argument names as listed above.
{no_alias_line}
- document_id not doc_id. max_length not max_words. channel not chan.
- JSON only."""

    def build_user_message(self, task_text: str) -> str:
        return task_text

