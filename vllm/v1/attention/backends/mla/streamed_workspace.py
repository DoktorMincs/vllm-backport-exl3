# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental one-window gather allocation with request-level chunking.

Only capacity and request partitioning change. Never use a compressed coordinate
budget larger than the actual compressed gather allocation.
"""
def eligible(config):
    text=getattr(config.model_config,'hf_text_config',None)
    parallel=config.parallel_config
    checks={
        'text_config':text is not None,
        'model_type':getattr(text,'model_type',None) in ('deepseek_v41','deepseek_v41_text'),
        'hidden_size==5120':getattr(text,'hidden_size',None)==5120,
        'layers==40':getattr(text,'num_hidden_layers',None)==40,
        'experts==384':getattr(text,'n_routed_experts',None)==384,
        'compress_ratios':bool(getattr(text,'compress_ratios',None)) and all(r in (0,1,2) for r in text.compress_ratios),
        'block_size==128':config.cache_config.block_size==128,
        'dcp==1':parallel.decode_context_parallel_size==1,
        'pcp==1':parallel.prefill_context_parallel_size==1,
    }
    ok=all(checks.values())
    if not ok:
        from vllm.logger import init_logger
        init_logger(__name__).info_once(
            'streamed indexer workspace NOT eligible (falls back to max_model_len*40 rows): failed=%s '
            '(model_type=%r block_size=%r)',
            [k for k,v in checks.items() if not v],
            getattr(text,'model_type',None), config.cache_config.block_size)
    return ok


def gather_capacity(config):
    length=int(config.model_config.max_model_len)
    if not eligible(config):return length*40
    drafts=int(config.num_speculative_tokens)
    if drafts<0:raise ValueError('Negative draft count')
    grain=128*max(1,max(config.model_config.hf_text_config.compress_ratios))
    return ((length+drafts+1+grain-1)//grain)*grain


def chunk_capacity(config,uncompressed_capacity,ratio):
    if not eligible(config):return uncompressed_capacity
    if ratio not in (1,2):raise ValueError('Unqualified indexer compression ratio')
    return uncompressed_capacity//ratio
