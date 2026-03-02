"""
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
"""
from typing import Tuple

import torch
import stream_kernels

__all__ = ['fused_rope_qk_apply']


@torch.library.custom_op('steam::fused_rope_qk_forward', mutates_args=(), device_types='cuda')
def _fused_rope_qk_forward(
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        freq: torch.Tensor,
        grid: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''_fused_rope_qk_forward'''
    q_dst = torch.empty_like(q_src, dtype=dtype)
    k_dst = torch.empty_like(k_src, dtype=dtype)
    stream_kernels.fused_rope_qk_fwd(
        q_src, k_src, freq, grid, q_dst, k_dst,
    )
    return q_dst, k_dst


@_fused_rope_qk_forward.register_fake
def _fused_rope_qk_forward_fake(
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        freq: torch.Tensor,
        grid: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''_fused_rope_qk_forward_fake'''
    return torch.empty_like(q_src, dtype=dtype), torch.empty_like(k_src, dtype=dtype)


def fused_rope_qk_apply(
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        grid_sizes: torch.Tensor,
        freq: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''fused_rope_qk_apply'''
    assert q_src.size(0) == grid_sizes.size(0)
    if grid_sizes.dtype != torch.int64:
        grid_sizes = grid_sizes.to(dtype=torch.int64)
    if grid_sizes.device != q_src.device:
        grid_sizes = grid_sizes.to(device=q_src.device)
    if torch.__version__ >= '2.4.0':
        return torch.ops.steam.fused_rope_qk_forward(
            q_src, k_src, freq, grid_sizes, dtype,
        )
    else:
        return _fused_rope_qk_forward(q_src, k_src, freq, grid_sizes, dtype)
