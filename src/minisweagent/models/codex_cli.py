"""Local Codex CLI-backed model.

This backend treats the local ``codex`` executable as a text-only model
provider for GEAK. GEAK remains responsible for executing tools/actions.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS


def _default_codex_bin() -> str:
    return os.getenv("CODEX_BIN", "codex")


def _default_timeout() -> float:
    return float(os.getenv("CODEX_CLI_TIMEOUT", "600"))


@dataclass
class CodexCliModelConfig:
    model_name: str = "codex-cli"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    codex_bin: str = field(default_factory=_default_codex_bin)
    cwd: str | None = field(default_factory=lambda: os.getenv("CODEX_CLI_CWD"))
    profile: str | None = None
    sandbox: str = "read-only"
    timeout: float = field(default_factory=_default_timeout)
    cost_per_call: float = 0.0
    ephemeral: bool = True
    skip_git_repo_check: bool = True
    ignore_rules: bool = False
    ignore_user_config: bool = False
    oss: bool = False
    local_provider: str | None = None
    config_overrides: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    prompt_preamble: str = (
        "You are the next-message language model backend for GEAK. "
        "Return only the assistant message that should be added to the conversation. "
        "GEAK, not Codex CLI, is responsible for executing commands and tools. "
        "If the next assistant message should call a GEAK tool, return exactly one JSON object "
        'and no prose: {"tool_call":{"function":{"name":"TOOL_NAME","arguments":{}}}}. '
        "Otherwise return the assistant text normally."
    )


_CODEX_DEFAULT_MODEL_SENTINELS = frozenset({"", "codex", "codex-cli", "codex_cli", "local-codex", "default"})


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise TypeError(f"Expected string or list of strings, got {type(value).__name__}")


def _merge_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
    retained_model_kwargs: dict[str, Any] = {}
    config_fields = set(CodexCliModelConfig.__dataclass_fields__) - {"model_kwargs"}
    for key, value in model_kwargs.items():
        if key in config_fields:
            kwargs.setdefault(key, value)
        else:
            # Keep provider-specific leftovers (for example AMD/OpenAI
            # temperature/max_tokens/reasoning settings) for template/debug
            # visibility, but do not pass them as CodexCliModelConfig fields.
            retained_model_kwargs[key] = value
    kwargs["model_kwargs"] = retained_model_kwargs
    if "config_overrides" in kwargs:
        kwargs["config_overrides"] = _coerce_str_list(kwargs["config_overrides"])
    if "extra_args" in kwargs:
        kwargs["extra_args"] = _coerce_str_list(kwargs["extra_args"])

    retained_top_level: dict[str, Any] = {}
    for key in list(kwargs):
        if key not in config_fields and key != "model_kwargs":
            retained_top_level[key] = kwargs.pop(key)
    retained_model_kwargs.update(retained_top_level)
    kwargs["model_kwargs"] = retained_model_kwargs
    return kwargs


def _format_tool_call(tool_call: Any) -> str:
    try:
        return json.dumps(tool_call, ensure_ascii=False)
    except TypeError:
        return str(tool_call)


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Return JSON objects embedded in *text*.

    Codex CLI is a text interface, so tool calls must be represented as text.
    Accept either a pure JSON response or a fenced/embedded JSON object.
    """
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.S))

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                objects.append(loaded)
                continue
        except json.JSONDecodeError:
            pass

        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                loaded, _end = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                objects.append(loaded)
                break
    return objects


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalise_text_tool_call(payload: dict[str, Any], call_index: int = 0) -> dict[str, Any] | None:
    raw: Any = payload
    for key in ("tool_call", "tool", "function_call"):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break

    if not isinstance(raw, dict):
        return None

    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", raw.get("arguments", raw.get("args", {})))
    else:
        name = raw.get("name")
        arguments = raw.get("arguments", raw.get("args", {}))

    if not name:
        return None

    return {
        "id": str(raw.get("id") or f"call_codex_{call_index}"),
        "function": {
            "name": str(name),
            "arguments": _coerce_tool_arguments(arguments),
        },
    }


def extract_text_tool_call(content: str, call_index: int = 0) -> dict[str, Any] | None:
    for obj in _extract_json_objects(content):
        tool_call = _normalise_text_tool_call(obj, call_index=call_index)
        if tool_call is not None:
            return tool_call
    return None


def extract_shell_tool_call(content: str, call_index: int = 0) -> dict[str, Any] | None:
    """Convert a single shell fenced block into GEAK's bash tool call."""
    matches = re.findall(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", content, re.DOTALL)
    if len(matches) != 1:
        return None
    return {
        "id": f"call_codex_{call_index}",
        "function": {
            "name": "bash",
            "arguments": {"command": matches[0].strip()},
        },
    }


def format_messages_for_codex(messages: list[dict[str, Any]]) -> str:
    """Render GEAK chat messages into a single Codex exec prompt."""
    blocks: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        role = str(msg.get("role", "user")).upper()
        content = msg.get("content", "")
        parts = [f"### Message {idx}: {role}", str(content)]
        if msg.get("tool_calls"):
            parts.append("<tool_call>")
            parts.append(_format_tool_call(msg["tool_calls"]))
            parts.append("</tool_call>")
        if role == "TOOL":
            name = msg.get("name", "")
            call_id = msg.get("tool_call_id", "")
            parts.insert(1, f"Tool name: {name}\nTool call id: {call_id}")
        blocks.append("\n".join(parts).strip())
    return "\n\n".join(blocks)


class CodexCliModel:
    """Use ``codex exec`` as a local GEAK model backend."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = CodexCliModelConfig(**_merge_model_kwargs(kwargs))
        self.cost = 0.0
        self.n_calls = 0
        self.tools: list[dict[str, Any]] = []

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        self.tools = list(tools)

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        prompt_parts = [self.config.prompt_preamble]
        if self.tools:
            tool_lines = []
            tool_names = set()
            for raw in self.tools:
                func = raw.get("function", raw)
                name = func.get("name")
                if name:
                    tool_names.add(str(name))
                    desc = str(func.get("description", "")).strip()
                    params = func.get("parameters") or func.get("input_schema") or {}
                    try:
                        params_text = json.dumps(params, ensure_ascii=False)
                    except TypeError:
                        params_text = str(params)
                    line = f"- {name}"
                    if desc:
                        line += f": {desc}"
                    if params_text and params_text != "{}":
                        line += f" Parameters: {params_text}"
                    tool_lines.append(line)
            if tool_lines:
                prompt_parts.append(
                    "GEAK tools available to the outer agent. When you need to use one of these tools, "
                    "return exactly one JSON object and no markdown/prose: "
                    '{"tool_call":{"function":{"name":"TOOL_NAME","arguments":{...}}}}. '
                    "This JSON tool-call protocol overrides any conversation text that asks for a "
                    "triple-backtick shell command. For shell work, call the `bash` tool through JSON.\n"
                    + "\n".join(tool_lines)
                )
                if "save_and_test" in tool_names:
                    prompt_parts.append(
                        "Optimization-agent rule: do not call `submit` and do not emit "
                        "`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` until you have edited the target code "
                        "and run `save_and_test` at least once. If no useful optimization works, still "
                        "record the best attempted patch with `save_and_test` before submitting."
                    )
        prompt_parts.append("Conversation:")
        prompt_parts.append(format_messages_for_codex(messages))
        return "\n\n".join(part for part in prompt_parts if part)

    def _build_command(self, output_path: Path) -> list[str]:
        cmd = [self.config.codex_bin, "exec"]
        if self.config.model_name not in _CODEX_DEFAULT_MODEL_SENTINELS:
            cmd.extend(["--model", self.config.model_name])
        if self.config.profile:
            cmd.extend(["--profile", self.config.profile])
        if self.config.sandbox:
            cmd.extend(["--sandbox", self.config.sandbox])
        if self.config.cwd:
            cmd.extend(["--cd", self.config.cwd])
        if self.config.oss:
            cmd.append("--oss")
        if self.config.local_provider:
            cmd.extend(["--local-provider", self.config.local_provider])
        if self.config.ephemeral:
            cmd.append("--ephemeral")
        if self.config.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if self.config.ignore_rules:
            cmd.append("--ignore-rules")
        if self.config.ignore_user_config:
            cmd.append("--ignore-user-config")
        for override in self.config.config_overrides:
            cmd.extend(["--config", override])
        cmd.extend(["--color", "never", "--output-last-message", str(output_path)])
        cmd.extend(self.config.extra_args)
        cmd.append("-")
        return cmd

    def _run_codex(self, prompt: str) -> tuple[str, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="geak-codex-") as tmpdir:
            output_path = Path(tmpdir) / "last_message.txt"
            cmd = self._build_command(output_path)
            env = os.environ.copy()
            env.setdefault("NO_COLOR", "1")
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.config.timeout,
                check=False,
                env=env,
            )
            output = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else proc.stdout.strip()
            extra = {
                "command": shlex.join(cmd[:-1] + ["<stdin>"]),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
            if proc.returncode != 0:
                raise RuntimeError(
                    "codex exec failed "
                    f"(returncode={proc.returncode}). stderr:\n{proc.stderr[-4000:]}"
                )
            return output, extra

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise TypeError(f"CodexCliModel.query does not support per-call kwargs yet: {sorted(kwargs)}")
        content, extra = self._run_codex(self._build_prompt(messages))
        tool_call = extract_text_tool_call(content, call_index=self.n_calls + 1)
        tool_names = {
            str(raw.get("function", raw).get("name"))
            for raw in self.tools
            if raw.get("function", raw).get("name")
        }
        if tool_call is None and "bash" in tool_names:
            tool_call = extract_shell_tool_call(content, call_index=self.n_calls + 1)
        self.n_calls += 1
        self.cost += self.config.cost_per_call
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return {
            "content": "" if tool_call else content,
            "tools": tool_call or "",
            "extra": {"response": extra | {"raw_content": content}},
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
