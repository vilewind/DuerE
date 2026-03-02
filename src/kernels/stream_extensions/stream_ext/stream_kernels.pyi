"""
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
"""
import torch
from typing import List


def fused_rope_qk_fwd(
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        freq: torch.Tensor,
        grid: torch.Tensor,
        q_dst: torch.Tensor,
        k_dst: torch.Tensor,
) -> List[torch.Tensor]:
    ...
