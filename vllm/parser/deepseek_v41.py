# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls."""

import functools
import os
from dataclasses import replace

import regex as re

from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END as _V4_INVOKE_END,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_PREFIX as _V4_INVOKE_PREFIX,
)
from vllm.parser.deepseek_v4 import (
    DSML_PARAM_CLOSE as _V4_PARAM_CLOSE,
)
from vllm.parser.deepseek_v4 import (
    DSML_PARAM_START as _V4_PARAM_START,
)
from vllm.parser.deepseek_v4 import (
    DSML_TOOL_END as _V4_TOOL_END,
)
from vllm.parser.deepseek_v4 import (
    DSML_TOOL_START as _V4_TOOL_START,
)
from vllm.parser.deepseek_v4 import (
    DSML_TOOL_START_VARIANTS as _V4_TOOL_START_VARIANTS,
)
from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _repair_unclosed_param,
    deepseek_v4_config,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

# Spelling variants of ``DSML_TOOL_START`` tolerated on input. The checkpoint
# drops the separating space, and at long context falls back to the unspaced V4
# wrapper, so the V4 spellings are accepted too. First entry of each terminal's
# tuple stays the canonical one that ``terminal_literal`` reports.
DSML_TOOL_START_VARIANTS: tuple[str, ...] = (
    "<｜DSML｜calls>",
    "<｜DSML｜ call>",
    _V4_TOOL_START,
    *_V4_TOOL_START_VARIANTS,
)

_CANONICAL_TERMINALS = {
    "TOOL_START": DSML_TOOL_START,
    "TOOL_END": DSML_TOOL_END,
    "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
    "INVOKE_END": DSML_INVOKE_END,
    "PARAM_START": DSML_PARAM_START,
    "PARAM_CLOSE": DSML_PARAM_CLOSE,
}

# Tolerated alternatives per terminal. Accepting a V4 wrapper opener is only
# useful if the rest of the block parses too, so every terminal that delimits
# the payload carries its unspaced V4 spelling as well.
_TOLERATED_VARIANTS: dict[str, tuple[str, ...]] = {
    "TOOL_START": DSML_TOOL_START_VARIANTS,
    "TOOL_END": (_V4_TOOL_END,),
    "INVOKE_PREFIX": (_V4_INVOKE_PREFIX,),
    "INVOKE_END": (_V4_INVOKE_END,),
    "PARAM_START": (_V4_PARAM_START,),
    "PARAM_CLOSE": (_V4_PARAM_CLOSE,),
}

# The space after the marker is the only difference between the two families,
# and it is the part that gets dropped, so parameter patterns accept it
# optionally rather than pinning one spelling.
_PARAM_OPEN = r"<｜DSML｜ ?parameter"
_PARAM_SHUT = r"</｜DSML｜ ?parameter>"

_PARAM_RE = re.compile(
    rf'{_PARAM_OPEN}\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*?)"
    rf"(?:{_PARAM_SHUT}|(?={_PARAM_OPEN}\s+name=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'{_PARAM_OPEN}\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*)$",
    re.DOTALL,
)
# Opener of a parameter whose closing tag never arrived; used only by the
# opt-in repair path.
_UNCLOSED_PARAM_RE = re.compile(
    rf'{_PARAM_OPEN}\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*)$",
    re.DOTALL,
)

_STRICT_ENV = "VLLM_DS41_STRICT_DSML"
_NOTE_ENV = "VLLM_DS41_MALFORMED_ARGS_NOTE"
_DEFAULT_NOTE = " [tool call omitted: malformed DSML]"


def _strict_dsml_enabled() -> bool:
    return os.environ.get(_STRICT_ENV, "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _malformed_args_note() -> str:
    note = os.environ.get(_NOTE_ENV)
    if note is None:
        return _DEFAULT_NOTE
    # The note must never be able to start a DSML marker, or the client's
    # text recoverer could rebuild a call out of it.
    return note.replace("<", "")


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    # Replacing ``terminals`` wholesale would otherwise drop the variant
    # tuples deepseek_v4_config declares, leaving one accepted spelling each.
    terminal_overrides = {
        name: (canon, *_TOLERATED_VARIANTS.get(name, ()))
        for name, canon in _CANONICAL_TERMINALS.items()
    }
    strict = _strict_dsml_enabled()
    return replace(
        config,
        name="deepseek_v41",
        terminals={**config.terminals, **terminal_overrides},
        # Token-id matching has no variant form: a special token is either in
        # the vocab or it is not, so keep the canonical spelling here.
        token_id_terminals={
            key: _CANONICAL_TERMINALS.get(key, value)
            for key, value in config.token_id_terminals.items()
        },
        arg_converter=functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
            strict_headers=strict,
        ),
        # Opt-in containment.  Everything below is inert by default so the
        # V4.1 config keeps behaving exactly as before unless the operator
        # sets VLLM_DS41_STRICT_DSML=1.
        defer_tool_deltas=strict,
        on_malformed_args="content" if strict else "call",
        malformed_args_note=_malformed_args_note() if strict else "",
        validate_tool_names=strict,
        validate_required_args=strict,
        repair_unclosed_param=(
            functools.partial(
                _repair_unclosed_param,
                param_re=_PARAM_RE,
                open_re=_UNCLOSED_PARAM_RE,
                close=DSML_PARAM_CLOSE,
            )
            if strict
            else None
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)
