# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.config.model import _apply_derived_max_len_cap


@pytest.mark.parametrize(
    "env,value,expected",
    [
        (None, 1048576, 1048576),
        ("0", 1048576, 1048576),
        ("262144", 1048576, 262144),
        ("262144", 100000, 100000),
        ("not-a-number", 1048576, 1048576),
    ],
)
def test_derived_max_len_cap(monkeypatch, env, value, expected):
    if env is None:
        monkeypatch.delenv("VLLM_MAX_MODEL_LEN_CAP", raising=False)
    else:
        monkeypatch.setenv("VLLM_MAX_MODEL_LEN_CAP", env)
    assert _apply_derived_max_len_cap(value) == expected
