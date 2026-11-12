# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in strict DSML containment for DeepSeek V4.1.

``VLLM_DS41_STRICT_DSML`` is the only switch that changes behaviour, and it
must leave every other parser (and this one, by default) untouched.
"""

import pytest

from vllm.parser.deepseek_v41 import (
    DSML_PARAM_CLOSE,
    _PARAM_RE,
    _UNCLOSED_PARAM_RE,
    _repair_unclosed_param,
    deepseek_v41_config,
)

_STRICT_ENV = "VLLM_DS41_STRICT_DSML"
_NOTE_ENV = "VLLM_DS41_MALFORMED_ARGS_NOTE"


@pytest.fixture
def config_env(monkeypatch):
    def _apply(**env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        deepseek_v41_config.cache_clear()

    yield _apply
    deepseek_v41_config.cache_clear()


@pytest.mark.parametrize("value", [None, "0", "false", "no", "off", ""])
def test_strict_mode_is_off_unless_asked_for(config_env, value):
    config_env(**{_STRICT_ENV: value})
    config = deepseek_v41_config()
    assert config.defer_tool_deltas is False
    assert config.on_malformed_args == "call"
    assert config.validate_tool_names is False
    assert config.validate_required_args is False
    assert config.repair_unclosed_param is None
    assert config.malformed_args_note == ""


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_strict_mode_enables_containment(config_env, value):
    config_env(**{_STRICT_ENV: value})
    config = deepseek_v41_config()
    assert config.defer_tool_deltas is True
    assert config.on_malformed_args == "content"
    assert config.validate_tool_names is True
    assert config.validate_required_args is True
    assert config.repair_unclosed_param is not None
    assert config.malformed_args_note
    assert "<" not in config.malformed_args_note


def test_note_override_cannot_start_a_marker(config_env):
    config_env(**{_STRICT_ENV: "1", _NOTE_ENV: "｜DSML｜ omitted>"})
    note = deepseek_v41_config().malformed_args_note
    assert "<" not in note
    assert "omitted" in note


def test_strict_mode_does_not_leak_into_v4(config_env):
    from vllm.parser.deepseek_v4 import deepseek_v4_config

    config_env(**{_STRICT_ENV: "1"})
    v4 = deepseek_v4_config()
    assert v4.defer_tool_deltas is False
    assert v4.on_malformed_args == "call"
    assert v4.validate_tool_names is False
    assert v4.repair_unclosed_param is None


# ── Trailing-parameter repair ────────────────────────────────────────
#
# The generation can stop after a parameter's value, before its closing
# tag.  Closing it is only safe when the name is a property the tool
# declares and the ``string`` flag agrees with the declared type.


def _repair(body, properties):
    return _repair_unclosed_param(
        body,
        properties,
        param_re=_PARAM_RE,
        open_re=_UNCLOSED_PARAM_RE,
        close=DSML_PARAM_CLOSE,
    )


def test_repair_closes_a_trailing_parameter_after_a_newline():
    body = '\n<｜DSML｜parameter name="path" string="true">/etc/hostname'
    assert _repair(body, {"path": {"type": "string"}}) == body + DSML_PARAM_CLOSE


def test_repair_keeps_the_already_closed_prefix():
    body = (
        '<｜DSML｜parameter name="path" string="true">/a</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="content" string="true">b'
    )
    properties = {"path": {"type": "string"}, "content": {"type": "string"}}
    assert _repair(body, properties) == body + DSML_PARAM_CLOSE


def test_repair_refuses_an_undeclared_property():
    body = '\n<｜DSML｜parameter name="evil" string="true">x'
    assert _repair(body, {"path": {"type": "string"}}) is None


def test_repair_refuses_a_type_mismatch():
    body = '\n<｜DSML｜parameter name="count" string="true">5'
    assert _repair(body, {"count": {"type": "integer"}}) is None


def test_repair_refuses_a_value_that_still_carries_the_marker():
    body = '\n<｜DSML｜parameter name="path" string="true">a<｜DSML｜parameterx b'
    assert _repair(body, {"path": {"type": "string"}}) is None


def test_repair_accepts_a_number_for_a_number_property():
    body = '\n<｜DSML｜parameter name="count" string="false">5'
    assert _repair(body, {"count": {"type": "integer"}}) == body + DSML_PARAM_CLOSE
