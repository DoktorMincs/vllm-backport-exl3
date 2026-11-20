# SPDX-License-Identifier: Apache-2.0
"""VLLM_SKIP_IDLE_STAGED_WRITES: UVA copies only after add_request."""

import pytest

from vllm.v1.worker.gpu.sample import sampler as sampler_mod


class _State:
    def __init__(self):
        self.applied = 0

    def apply_staged_writes(self):
        self.applied += 1

    def add_request(self, *args, **kwargs):
        pass


def _make():
    s = object.__new__(sampler_mod.Sampler)
    for name in (
        "sampling_states",
        "penalties_state",
        "logit_bias_state",
        "bad_words_state",
        "logprob_token_ids_state",
        "thinking_budget_state",
    ):
        setattr(s, name, _State())
    s.trace_replay_state = None
    s._staged_dirty = True
    return s


@pytest.fixture(autouse=True)
def _restore():
    old = sampler_mod._SKIP_IDLE_STAGED_WRITES
    yield
    sampler_mod._SKIP_IDLE_STAGED_WRITES = old


def test_default_applies_every_step():
    sampler_mod._SKIP_IDLE_STAGED_WRITES = False
    s = _make()
    for _ in range(3):
        s.apply_staged_writes()
    assert s.sampling_states.applied == 3
    assert s.thinking_budget_state.applied == 3


def test_skip_idle_copies_only_after_a_request():
    sampler_mod._SKIP_IDLE_STAGED_WRITES = True
    s = _make()
    s.apply_staged_writes()  # initial state is dirty: applied once
    s.apply_staged_writes()
    s.apply_staged_writes()
    assert s.sampling_states.applied == 1
    assert s.logit_bias_state.applied == 1
    # thinking budget has its own guard and keeps being called
    assert s.thinking_budget_state.applied == 3
    s._staged_dirty = True  # what add_request does
    s.apply_staged_writes()
    assert s.sampling_states.applied == 2
    assert s._staged_dirty is False
