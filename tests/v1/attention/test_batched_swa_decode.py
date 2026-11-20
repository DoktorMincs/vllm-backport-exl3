from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.mark.parametrize("pad", [0, 5])
def test_batched_matches_per_group(pad):
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
        prebuild_swa_decode_batched,
    )

    torch.manual_seed(1)
    dev = "cuda"
    G, R, Q, W, BS, MAXB, MAXT = 4, 3, 6, 128, 64, 40, 64
    seq = torch.tensor([300, 70, 2000], dtype=torch.int32, device=dev)
    qsl = torch.arange(R + 1, dtype=torch.int32, device=dev) * Q
    n = R * Q
    t2r = torch.repeat_interleave(torch.arange(R, dtype=torch.int32, device=dev), Q)
    t2r = torch.cat([t2r, torch.zeros(MAXT - n, dtype=torch.int32, device=dev)])
    bts = [
        torch.randint(1, 1000, (R, MAXB), dtype=torch.int32, device=dev)
        for _ in range(G)
    ]
    sms = []
    for _ in range(G):
        sm = torch.randint(0, 10000, (n + pad,), dtype=torch.int64, device=dev)
        if pad:
            sm[n:] = -1
        sm[4] = -1  # an invalid token inside the decode range
        sms.append(sm)

    def mk():
        return SimpleNamespace(
            window_size=W, block_size=BS, decode_threshold=Q, device=dev,
            decode_swa_indices=torch.full((MAXT, 1, W), 9, dtype=torch.int32, device=dev),
            decode_swa_lens=torch.full((MAXT,), 9, dtype=torch.int32, device=dev),
            is_valid_token=torch.zeros(MAXT, dtype=torch.bool, device=dev),
        )

    new, ref = [mk() for _ in range(G)], [mk() for _ in range(G)]
    cm = SimpleNamespace(
        num_actual_tokens=n, max_query_len=Q, causal=True,
        query_start_loc=qsl, seq_lens=seq,
    )
    assert prebuild_swa_decode_batched(new, bts, sms, cm, t2r[:n])
    for b, bt, sm in zip(ref, bts, sms):
        b.decode_swa_lens[n:] = 0
        b.is_valid_token[: sm.shape[0]].copy_(sm >= 0)
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
            b.decode_swa_indices, b.decode_swa_lens, W, W,
            b.decode_swa_lens, b.decode_swa_lens, qsl, seq, t2r,
            b.is_valid_token, bt, BS, num_tokens=n, token_offset=0,
        )
    for a, b in zip(new, ref):
        assert torch.equal(a.decode_swa_indices[:n], b.decode_swa_indices[:n])
        assert torch.equal(a.decode_swa_lens, b.decode_swa_lens)
        assert torch.equal(a.is_valid_token[: n + pad], b.is_valid_token[: n + pad])
