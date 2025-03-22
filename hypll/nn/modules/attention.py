from typing import List, Optional, Union, Tuple
import math
import torch
from torch import nn
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
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
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
        need_weights: bool = False
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

        if any([s.tensor.isnan().any() for s in split_q]):
            raise ValueError(f"NaNs in projected Qs: {split_q}")
        if any([s.tensor.isnan().any() for s in split_k]):
            raise ValueError(f"NaNs in projected Ks: {split_k}")
        if any([s.tensor.isnan().any() for s in split_v]):
            raise ValueError(f"NaNs in projected Vs: {split_v}")

        # Apply similarity function f and activation g to Qs and Ks to obtain weights (g(f(q, k)))
        similarities = [
            self.manifold.attention_similarity(queries=q, keys=k) for q, k in zip(split_q, split_k)
        ]
        if any([s.isnan().any() for s in similarities]):
            raise ValueError(f"NaNs in similarities: {similarities}")
        weights = [
            self.manifold.attention_activation(s) for s in similarities
        ]
        if any([w.isnan().any() for w in weights]):
            raise ValueError(f"NaNs in weights: {weights}")

        # Aggregate values with sequence-aware specialization of the centroid
        aggregates = [
            self.manifold.attention_midpoint(x=v, w=w) for v, w in zip(split_v, weights)
        ]

        # Concatenate outputs
        result = self.manifold.cat(aggregates, dim=-1)

        # Handle non-batch-first output
        if not self.batch_first:
            result = result.transpose(0, 1)

        if need_weights:
            return result, weights
        else:
            return result


class HFullDimensionMultiHeadAttention(Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        use_separate_kv_per_head: bool = False,
    ):
        """ 
        Args:
            embed_dim: Dimension of the embedding space
            num_heads: Number of attention heads
            manifold: Manifold to use
            kdim: Dimension of the key space
            vdim: Dimension of the value space
            batch_first: If True, the first dimension is the batch dimension
            use_separate_kv_per_head: If True, use separate key and value for each head. 
                If False then only the queries differ per head, which is Multi-Query Attention.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.use_separate_kv_per_head = use_separate_kv_per_head
        self.batch_first = batch_first

        # Introduce a scaling factor to prevent gradients from becoming too small.
        self.scale_tau = nn.Parameter(torch.zeros(1))
        self.scale_gamma = nn.Parameter(torch.zeros(1))

        # Create separate projection layers for each head
        self.q_maps = nn.ModuleList([
            HLinear(embed_dim, embed_dim, manifold=manifold) 
            for _ in range(num_heads)
        ])
        if self.use_separate_kv_per_head:
            self.k_maps = nn.ModuleList([
                HLinear(embed_dim, embed_dim, manifold=manifold) 
                for _ in range(num_heads)
            ])
            self.v_maps = nn.ModuleList([
                HLinear(embed_dim, embed_dim, manifold=manifold) 
                for _ in range(num_heads)
            ])
        else:
            self.k_map = HLinear(embed_dim, embed_dim, manifold=manifold)
            self.v_map = HLinear(embed_dim, embed_dim, manifold=manifold)

        # For 2/a
        self.output_projection = HLinear(embed_dim*num_heads, embed_dim, manifold=manifold)
        # For 2/b
        # self.output_projection = HLinear(num_heads, 1, manifold=manifold)
        # self.output_projection = nn.Linear(num_heads, 1)
        
    def forward(self, query, key, value, need_weights=False):
        # Handle non-batch-first input
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        
        # Each head processes the full embedding
        head_outputs = []
        for i in range(self.num_heads):
            q_i = self.q_maps[i](query)
            k_i = self.k_maps[i](key) if self.use_separate_kv_per_head else self.k_map(key)
            v_i = self.v_maps[i](value) if self.use_separate_kv_per_head else self.v_map(value)
            
            # Calculate attention (no splitting)
            # The code of HNN++ has the following line for similarity calculation, which I tried to replicate:
            # x = - self.ball.dist_matmul(x, encoder_out[0]) / self.scale.exp()
            similarities = self.manifold.attention_similarity(queries=q_i, keys=k_i, tau=1/self.scale_tau.exp(), gamma=self.scale_gamma)
            weights = self.manifold.attention_activation(similarities)
            output_i = self.manifold.attention_midpoint(v_i, weights)
            
            head_outputs.append(output_i)
        
        # Combine outputs from different heads
        # Option 1: Average in hyperbolic space
        # This doesn't work now, need to figure out how to parameterize the midpoint calculation correctly
        # result = self.manifold.attention_midpoint(x=head_outputs, w=[1/self.num_heads]*self.num_heads)
        
        # Option 2/a: Simply concat across the embed dim and map back to embed dim.
        # This is simple, but it uses a concatenation along the manifold dimension, so it's quite expensive
        # and results in beta scaling. I wanted to avoid this with 2/b, but that version didn't work, while this one does.
        concat = self.manifold.cat(head_outputs, dim=-1)
        result = self.output_projection(concat)

        # Option 2/b: Project to higher dimension then back
        # I thought this'd be theoretically more correct, but doesn't work at all
        # I've tried it with HLinear layer, but that's obviously not correct, as it will always perform an operation
        # along the manifold dimension, which is not the new final dimension we created. Also this last dimension
        # does not contain points on the manifold.
        # I've also tried it with a Euclidean linear layer, but that also didnt' work.
        # head_outputs = [h.unsqueeze(-1) for h in head_outputs]
        # concat = self.manifold.cat(head_outputs, dim=-1)
        # result = self.output_projection(concat.tensor)
        # result = result.squeeze(-1)
        # result = ManifoldTensor(result, self.manifold)
        
        if not self.batch_first:
            result = result.transpose(0, 1)

        if need_weights:
            return result, weights
        else:
            return result
