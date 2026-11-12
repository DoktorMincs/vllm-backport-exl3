# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict DSML containment for DeepSeek V4.1.

The checkpoint corrupts DSML at long context: a parameter header can lose
``parameter``, an invoke name can be truncated, or the block can end
without its closing tags.  The legacy converter turned such a body into a
literal ``"{}"``, so the server announced a call the client could never
validate and the agent looped on the same failure.

With ``VLLM_DS41_STRICT_DSML=1`` a body that cannot be read becomes
content instead, and nothing is announced until the block closes.
"""

import json

import pytest

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    _test_request,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.tool_parsers.deepseekv41_engine_tool_parser import (
    DeepSeekV41EngineToolParser,
)

pytestmark = pytest.mark.skip_global_cleanup

D = "｜DSML｜"

TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    ),
]

TOOL_DICTS = [tool.model_dump() for tool in TOOLS]

CLEAN = (
    f"<{D}tool_calls>\n"
    f'<{D}invoke name="read">\n'
    f'<{D}parameter name="path" string="true">/etc/hostname</{D}parameter>\n'
    f"</{D}invoke>\n"
    f"</{D}tool_calls>"
)

# ``parameter`` dropped from the header: the converter reads no parameter
# and the legacy path announced ``{}``.
NO_PARAM_TAG = (
    f"<{D}tool_calls>\n"
    f'<{D}invoke name="read">\n'
    f'<{D} path" string="true">/etc/hostname</{D}parameter>\n'
    f"</{D}invoke>\n"
    f"</{D}tool_calls>"
)

# The generation was cut before the closing tags arrived.
BLOCK_UNCLOSED = (
    f"<{D}tool_calls>\n"
    f'<{D}invoke name="read">\n'
    f'<{D}parameter name="path" string="true">/etc/hostname'
)

# A well-formed header for a property the tool declares required, missing.
MISSING_REQUIRED = (
    f"<{D}tool_calls>\n"
    f'<{D}invoke name="read">\n'
    f'<{D}parameter name="limit" string="false">10</{D}parameter>\n'
    f"</{D}invoke>\n"
    f"</{D}tool_calls>"
)

FIRST_BAD_SECOND_GOOD = (
    f"<{D}tool_calls>\n"
    f'<{D}invoke name="read">\n'
    f'<{D} path" string="true">/etc/hostname</{D}parameter>\n'
    f"</{D}invoke>\n"
    f'<{D}invoke name="read">\n'
    f'<{D}parameter name="path" string="true">/etc/os-release</{D}parameter>\n'
    f"</{D}invoke>\n"
    f"</{D}tool_calls>"
)


@pytest.fixture
def strict(monkeypatch):
    """Turn the opt-in on for one test and restore the cached config."""
    monkeypatch.setenv("VLLM_DS41_STRICT_DSML", "1")
    deepseek_v41_config.cache_clear()
    yield
    deepseek_v41_config.cache_clear()


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.delenv("VLLM_DS41_STRICT_DSML", raising=False)
    deepseek_v41_config.cache_clear()
    yield
    deepseek_v41_config.cache_clear()


def make_parser() -> DeepSeekV41EngineToolParser:
    tokenizer = MockTokenizer({"": 1}, [])
    return DeepSeekV41EngineToolParser(tokenizer, tools=TOOLS)


def parse(text: str):
    parser = make_parser()
    result = parser.extract_tool_calls(text, _test_request(tools=TOOL_DICTS))
    return result


def stream(text: str, chunk: int = 7):
    parser = make_parser()
    request = _test_request(tools=TOOL_DICTS)
    deltas = []
    for i in range(0, len(text), chunk):
        delta = parser.extract_tool_calls_streaming(
            previous_text=text[:i],
            current_text=text[: i + chunk],
            delta_text=text[i : i + chunk],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
    tail = parser.finish_streaming()
    if tail is not None:
        deltas.append(tail)
    return deltas


def collect(deltas):
    """Fold deltas into ``(content, {index: call})``."""
    content = ""
    calls: dict[int, dict] = {}
    for delta in deltas:
        if delta.content:
            content += delta.content
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"name": None, "args": "", "id": None})
            if tc.id:
                call["id"] = tc.id
            if tc.function is not None:
                if tc.function.name:
                    call["name"] = tc.function.name
                if tc.function.arguments:
                    call["args"] += tc.function.arguments
    return content, calls


def test_clean_block_still_produces_a_call(strict):
    result = parse(CLEAN)
    assert result.tools_called
    assert [c.function.name for c in result.tool_calls] == ["read"]
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "path": "/etc/hostname"
    }


@pytest.mark.parametrize(
    "text",
    [NO_PARAM_TAG, MISSING_REQUIRED],
    ids=["no-param-tag", "missing-required"],
)
def test_unreadable_block_becomes_content(strict, text):
    result = parse(text)
    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content
    assert "DSML" in result.content


def test_rejected_body_note_cannot_start_a_marker(strict, monkeypatch):
    monkeypatch.setenv("VLLM_DS41_MALFORMED_ARGS_NOTE", f"<{D}omitted>")
    deepseek_v41_config.cache_clear()
    note = deepseek_v41_config().malformed_args_note
    assert "<" not in note
    result = parse(NO_PARAM_TAG)
    assert not result.tools_called
    assert result.content.endswith(note)


def test_unclosed_parameter_is_repaired(strict):
    """A trailing parameter that lost its closing tag is closed when its
    name and type match the tool schema."""
    result = parse(BLOCK_UNCLOSED)
    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "path": "/etc/hostname"
    }


def test_empty_body_respects_the_required_list(strict):
    """A call with no arguments is only legitimate when nothing is required."""
    required_tool = (
        f"<{D}tool_calls>\n"
        f'<{D}invoke name="read">\n'
        f"</{D}invoke>\n"
        f"</{D}tool_calls>"
    )
    result = parse(required_tool)
    assert not result.tools_called
    assert result.content

    free_tool = (
        f"<{D}tool_calls>\n"
        f'<{D}invoke name="ping">\n'
        f"</{D}invoke>\n"
        f"</{D}tool_calls>"
    )
    result = parse(free_tool)
    assert result.tools_called
    assert result.tool_calls[0].function.arguments == "{}"


def test_legacy_mode_keeps_the_empty_arguments_call(legacy):
    """Flags off must reproduce the pre-existing behaviour exactly."""
    result = parse(NO_PARAM_TAG)
    assert result.tools_called
    assert result.tool_calls[0].function.arguments == "{}"


def test_streaming_never_announces_a_rejected_call(strict):
    content, calls = collect(stream(NO_PARAM_TAG))
    assert calls == {}
    assert "DSML" in content


def test_streaming_publishes_a_clean_call_once(strict):
    content, calls = collect(stream(CLEAN))
    assert list(calls) == [0]
    assert calls[0]["name"] == "read"
    assert json.loads(calls[0]["args"]) == {"path": "/etc/hostname"}
    assert calls[0]["id"]
    assert "DSML" not in content


def test_streaming_reindexes_after_a_rejected_call(strict):
    """The surviving call keeps index 0 and matches the non-streaming value.

    A rejected first call must not leave a gap the client would read as a
    second, empty call.
    """
    content, calls = collect(stream(FIRST_BAD_SECOND_GOOD))
    assert list(calls) == [0]
    assert json.loads(calls[0]["args"]) == {"path": "/etc/os-release"}
    assert "DSML" in content

    result = parse(FIRST_BAD_SECOND_GOOD)
    assert [c.function.name for c in result.tool_calls] == ["read"]
    assert (
        result.tool_calls[0].function.arguments == calls[0]["args"]
    )


def test_streaming_matches_non_streaming_for_every_fixture(strict):
    for text in (CLEAN, NO_PARAM_TAG, BLOCK_UNCLOSED, MISSING_REQUIRED):
        _, calls = collect(stream(text))
        result = parse(text)
        assert [calls[i]["name"] for i in sorted(calls)] == [
            c.function.name for c in result.tool_calls
        ]
        assert [calls[i]["args"] for i in sorted(calls)] == [
            c.function.arguments for c in result.tool_calls
        ]
