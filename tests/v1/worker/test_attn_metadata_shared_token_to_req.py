# SPDX-License-Identifier: Apache-2.0
"""VLLM_SHARE_TOKEN_TO_REQ: KV groups reuse the first group's token->request map."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu import attn_utils


class _Builder:
    def __init__(self, max_tokens: int):
        self.buffer = torch.full((max_tokens,), -7, dtype=torch.int32)
        self.seen: list[torch.Tensor] = []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.seen.append(common_attn_metadata.token_to_req_indices(self.buffer))
        return object()


def _run(share: bool):
    attn_utils._SHARE_TOKEN_TO_REQ = share
    num_groups, num_reqs, per_req = 3, 3, 2
    num_tokens = num_reqs * per_req
    builders = [_Builder(16) for _ in range(num_groups)]
    attn_groups = [
        [SimpleNamespace(get_metadata_builder=lambda ub, b=b: b, layer_names=[f"l{i}"])]
        for i, b in enumerate(builders)
    ]
    qsl_cpu = torch.arange(0, (num_reqs + 1) * per_req, per_req, dtype=torch.int32)
    attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=qsl_cpu.clone(),
        query_start_loc_cpu=qsl_cpu,
        max_query_len=per_req,
        seq_lens=torch.full((num_reqs,), 10, dtype=torch.int32),
        max_seq_len=10,
        block_tables=[torch.zeros(num_reqs, 4, dtype=torch.int32)] * num_groups,
        slot_mappings=torch.zeros(num_groups, num_tokens, dtype=torch.int64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()] * num_groups),
    )
    return builders


@pytest.fixture(autouse=True)
def _restore():
    old = attn_utils._SHARE_TOKEN_TO_REQ
    yield
    attn_utils._SHARE_TOKEN_TO_REQ = old


def test_default_each_group_builds_its_own():
    builders = _run(False)
    ptrs = {b.seen[0].data_ptr() for b in builders}
    assert len(ptrs) == 3
    for b in builders:
        assert b.seen[0].tolist() == [0, 0, 1, 1, 2, 2]


def test_shared_reuses_first_group_result():
    builders = _run(True)
    first = builders[0].seen[0]
    assert first.tolist() == [0, 0, 1, 1, 2, 2]
    for b in builders[1:]:
        assert b.seen[0].data_ptr() == first.data_ptr()
        assert b.seen[0].tolist() == first.tolist()
        # the later builders never touched their own buffers
        assert (b.buffer == -7).all()
