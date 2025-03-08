from typing import List, Optional, Union, Tuple

import torch
from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.nn import HLinear
from hypll.tensors import ManifoldTensor


class HMultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        kdim: Optional[int],
        vdim: Optional[int],
        batch_first: bool = False,
    ):
        super(HMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim,
            "embed_dim must be divisible by num_heads"
        )

        self.q_map = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
        )
        self.k_map = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
        )
        self.v_map = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
        )

    def forward(
        self,
        query: ManifoldTensor, # [B, L_t, D] if batch_first
        key: ManifoldTensor, # [B, L_s, D] if batch_first
        value: ManifoldTensor, # [B, L_s, D] if batch_first
        return_weights: bool = False
    ) -> Union[ManifoldTensor, Tuple[ManifoldTensor, List[torch.Tensor]]]:
        # Handle non-batch-first input
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        # Project Q, K and V with FC-layers ([B, L_s, D] -> [B, L_s, hd], with hd = D, so not really a projection)
        proj_q = self.q_map(query)
        proj_k = self.k_map(key)
        proj_v = self.v_map(value)

        # Split projected Q, K and V into seperate Q, K and V per head (Q: [B, L_t, hd] -> h * [B, L_t, d])
        # Probably want to keep Qs as a tensor [h, B, L_t, d] to allow easy parallel compute
        split_q = proj_q.split(split_size_or_sections=self.head_dim, dim=-1)
        split_k = proj_k.split(split_size_or_sections=self.head_dim, dim=-1)
        split_v = proj_v.split(split_size_or_sections=self.head_dim, dim=-1)

        # Apply similarity function f and activation g to Qs and Ks to obtain weights (g(f(q, k)))
        similarities = [
            self.manifold.attention_similarity(queries=q, keys=k) for q, k in zip(split_q, split_k)
        ]
        weights = [
            self.manifold.attention_activation(s) for s in similarities
        ]

        # Aggregate values with sequence-aware specialization of the centroid
        aggregates = [
            self.manifold.attention_midpoint(x=v, w=w) for v, w in zip(split_v, weights)
        ]

        # Concatenate outputs
        result = self.manifold.cat(aggregates, dim=-1)

        # Handle non-batch-first output
        if not self.batch_first:
            result = result.transpose(0, 1)

        if return_weights:
            return result, weights
        else:
            return result
