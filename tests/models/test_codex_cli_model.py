from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.models import get_model
from minisweagent.models.codex_cli import (
    CodexCliModel,
    extract_shell_tool_call,
    extract_text_tool_call,
    format_messages_for_codex,
)


def test_format_messages_for_codex_includes_roles_tool_calls_and_tool_results():
    prompt = format_messages_for_codex([
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "I will call a tool",
            "tool_calls": {"id": "c1", "function": {"name": "bash", "arguments": {"command": "ls"}}},
        },
        {"role": "tool", "name": "bash", "tool_call_id": "c1", "content": "ok"},
    ])

    assert "### Message 1: SYSTEM" in prompt
    assert "sys" in prompt
    assert "<tool_call>" in prompt
    assert '"name": "bash"' in prompt
    assert "Tool name: bash" in prompt
    assert "Tool call id: c1" in prompt


def test_build_command_defaults_are_text_backend_safe(tmp_path):
    model = CodexCliModel(model_name="gpt-5", codex_bin="codex", cwd="/repo")

    cmd = model._build_command(tmp_path / "out.txt")

    assert cmd[:2] == ["codex", "exec"]
    assert ["--model", "gpt-5"] == cmd[2:4]
    assert "--ephemeral" in cmd
    assert "--skip-git-repo-check" in cmd
    assert ["--sandbox", "read-only"] == cmd[cmd.index("--sandbox") : cmd.index("--sandbox") + 2]
    assert ["--cd", "/repo"] == cmd[cmd.index("--cd") : cmd.index("--cd") + 2]
    assert cmd[-1] == "-"


def test_build_command_can_use_codex_cli_default_model(tmp_path):
    model = CodexCliModel(model_name="codex-cli", codex_bin="codex")

    cmd = model._build_command(tmp_path / "out.txt")

    assert "--model" not in cmd


def test_query_reads_output_last_message_and_updates_stats(tmp_path):
    def fake_run(cmd, input, text, capture_output, timeout, check, env):  # noqa: A002
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text("```bash\necho hi\n```", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ignored stdout", stderr="")

    model = CodexCliModel(codex_bin="codex", model_name="gpt-5-codex", cost_per_call=0.25)

    with patch("minisweagent.models.codex_cli.subprocess.run", side_effect=fake_run) as run:
        result = model.query([{"role": "user", "content": "say hi"}])

    assert result["content"] == "```bash\necho hi\n```"
    assert result["tools"] == ""
    assert model.n_calls == 1
    assert model.cost == 0.25
    assert run.call_args.kwargs["input"].startswith(model.config.prompt_preamble)


def test_extract_text_tool_call_accepts_json_object():
    tool_call = extract_text_tool_call(
        '{"tool_call":{"function":{"name":"generate_tasks","arguments":{"round_num":1}}}}',
        call_index=7,
    )

    assert tool_call == {
        "id": "call_codex_7",
        "function": {"name": "generate_tasks", "arguments": {"round_num": 1}},
    }


def test_extract_text_tool_call_accepts_fenced_tool_shape():
    tool_call = extract_text_tool_call(
        '```json\n{"tool":{"name":"dispatch_tasks","arguments":"{\\"task_paths\\":[\\"a.md\\"]}"}}\n```',
        call_index=2,
    )

    assert tool_call == {
        "id": "call_codex_2",
        "function": {"name": "dispatch_tasks", "arguments": {"task_paths": ["a.md"]}},
    }


def test_query_converts_text_tool_call_to_model_tool(tmp_path):
    def fake_run(cmd, input, text, capture_output, timeout, check, env):  # noqa: A002
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text('{"tool":{"name":"collect_results","arguments":{}}}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    model = CodexCliModel(codex_bin="codex")

    with patch("minisweagent.models.codex_cli.subprocess.run", side_effect=fake_run):
        result = model.query([{"role": "user", "content": "collect"}])

    assert result["content"] == ""
    assert result["tools"]["function"]["name"] == "collect_results"
    assert result["extra"]["response"]["raw_content"] == '{"tool":{"name":"collect_results","arguments":{}}}'


def test_extract_shell_tool_call_accepts_single_sh_fence():
    tool_call = extract_shell_tool_call("```sh\nls -la\n```", call_index=3)

    assert tool_call == {
        "id": "call_codex_3",
        "function": {"name": "bash", "arguments": {"command": "ls -la"}},
    }


def test_query_converts_shell_fence_to_bash_tool_when_bash_available(tmp_path):
    def fake_run(cmd, input, text, capture_output, timeout, check, env):  # noqa: A002
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text("```sh\nls\n```", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    model = CodexCliModel(codex_bin="codex")
    model.set_tools([{"name": "bash", "parameters": {}}])

    with patch("minisweagent.models.codex_cli.subprocess.run", side_effect=fake_run):
        result = model.query([{"role": "user", "content": "list files"}])

    assert result["content"] == ""
    assert result["tools"]["function"] == {"name": "bash", "arguments": {"command": "ls"}}
    assert result["extra"]["response"]["raw_content"] == "```sh\nls\n```"


def test_build_prompt_tells_codex_tools_override_bash_markdown_protocol():
    model = CodexCliModel(codex_bin="codex")
    model.set_tools([
        {
            "name": "bash",
            "description": "Execute shell commands directly in bash.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
        {"name": "save_and_test", "description": "Save current changes as patch.", "parameters": {}},
        {"name": "submit", "description": "Submit final result.", "parameters": {}},
    ])

    prompt = model._build_prompt([{"role": "user", "content": "Use one shell command in triple backticks."}])

    assert "JSON tool-call protocol overrides" in prompt
    assert "call the `bash` tool through JSON" in prompt
    assert "run `save_and_test` at least once" in prompt


def test_query_raises_on_codex_failure():
    def fake_run(cmd, input, text, capture_output, timeout, check, env):  # noqa: A002
        return SimpleNamespace(returncode=2, stdout="", stderr="bad auth")

    model = CodexCliModel(codex_bin="codex")

    with patch("minisweagent.models.codex_cli.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="bad auth"):
            model.query([{"role": "user", "content": "hello"}])


def test_get_model_supports_codex_cli_shortcut():
    model = get_model("codex-cli", {"model_class": "codex_cli", "codex_bin": "codex"})

    assert isinstance(model, CodexCliModel)
    assert model.config.model_name == "codex-cli"


def test_codex_cli_ignores_leftover_gateway_model_kwargs():
    model = get_model(
        "codex-cli",
        {
            "model_class": "codex_cli",
            "api_key": "",
            "model_kwargs": {
                "temperature": 0.0,
                "max_tokens": 16000,
                "reasoning": {"effort": "high"},
                "cwd": "/repo",
            },
        },
    )

    assert isinstance(model, CodexCliModel)
    assert model.config.cwd == "/repo"
    assert model.config.model_kwargs["api_key"] == ""
    assert model.config.model_kwargs["temperature"] == 0.0
