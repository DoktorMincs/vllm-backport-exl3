import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_ragged_direct_matches_copy_path():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    torch.manual_seed(0)
    dev = "cuda"
    n, w = 24, 128
    idx = torch.randint(0, 5000, (n, w), dtype=torch.int32, device=dev)
    lens = torch.randint(0, w + 1, (n,), dtype=torch.int32, device=dev)
    ref_i, ref_p = build_ragged_indices_from_dense(idx, lens)
    buf_i = torch.full((64 * w,), -7, dtype=torch.int32, device=dev)
    buf_p = torch.full((65,), -7, dtype=torch.int32, device=dev)
    out_i, out_p = build_ragged_indices_from_dense(
        idx, lens, indices_out=buf_i[: n * w], indptr_out=buf_p[: n + 1]
    )
    assert out_p.data_ptr() == buf_p.data_ptr()
    assert out_i.data_ptr() == buf_i.data_ptr()
    assert torch.equal(out_p, ref_p)
    nnz = int(ref_p[-1])
    assert torch.equal(out_i[:nnz], ref_i[:nnz])
