from typing import List, Optional, Union, Tuple
import math
import torch
from torch import nn
from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.nn import HLinear
from hypll.tensors import ManifoldTensor

class IntermediateValueModule(nn.Module):
    """ Module to save intermediate values for debugging and analysis
    
    Intermediate values are values within a Module, not an output of a Module.
    These could not be easily saved by the usual hooks, so we use this module to save them.
    This is used for values such as similarities, weights, aggregates within attention. """
    def __init__(self, name, enabled=True):
        super().__init__()
        self.name = name
        self.enabled = enabled
        # Register a buffer for the saved tensor (not a parameter since we don't want it trained)
        self.register_buffer('saved_tensor', None)
        # Register a parameter for the gradient to be tracked
        self.register_parameter('saved_grad', nn.Parameter(torch.zeros(1), requires_grad=True))
        # Store the hook handle to remove it later
        self.hook_handle = None
        
    def forward(self, x):
        # If not enabled (either locally or globally), just pass through without doing anything
        if not self.enabled:
            return x
            
        # For ManifoldTensor, we need to work with the underlying tensor
        if isinstance(x, ManifoldTensor):
            tensor = x.tensor
        else:
            tensor = x
            
        # Save the tensor for later access
        self.saved_tensor = tensor.detach()
        
        # Remove previous hook if it exists
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        # Register a hook to capture gradients during backward pass
        if tensor.requires_grad:
            def hook(grad):
                # Clone the gradient to our parameter
                if self.saved_grad.shape != grad.shape:
                    # Resize parameter if needed
                    self.saved_grad = nn.Parameter(torch.zeros_like(grad), requires_grad=True)
                    self.saved_grad.grad = grad.detach().clone()
                else:
                    # Just copy the data
                    self.saved_grad.data.copy_(grad.detach())
                    self.saved_grad.grad = grad.detach().clone()
                return grad
            
            self.hook_handle = tensor.register_hook(hook)
            
        # Return the original input unchanged to maintain the computation graph
        return x
        
    # This is the key method to handle checkpoint loading properly
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Custom state dict loading to handle the dynamic saved_grad and saved_tensor.
        
        This method resizes saved_grad to match the size in the checkpoint before loading,
        which avoids size mismatch errors.
        """
        saved_tensor_key = prefix + 'saved_tensor'
        saved_grad_key = prefix + 'saved_grad'
        
        # Handle saved_grad - resize it to match the checkpoint if needed
        if saved_grad_key in state_dict:
            checkpoint_grad = state_dict[saved_grad_key]
            if self.saved_grad.shape != checkpoint_grad.shape:
                # Resize parameter to match checkpoint size
                self.saved_grad = nn.Parameter(torch.zeros_like(checkpoint_grad), requires_grad=True)
        
        # Handle saved_tensor - it's a buffer so we can just set it directly
        if saved_tensor_key in state_dict:
            checkpoint_tensor = state_dict[saved_tensor_key]
            if checkpoint_tensor is not None:
                self.saved_tensor = checkpoint_tensor.clone() if checkpoint_tensor is not None else None
                
        # Call the parent method with the original state dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

class HMultiheadAttention(Module):
    """ Multihead attention module for hyperbolic manifolds, using intermediate value modules for debugging """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        attention_activation: str = "exp",
        causal: bool = True,
        save_intermediate_values: bool = False,
        use_attention_projection: bool = False,
        mlp_init_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self.attention_activation = attention_activation
        self.causal = causal
        self.save_intermediate_values = save_intermediate_values
        self.use_attention_projection = use_attention_projection
        self.mlp_init_scale = mlp_init_scale

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

        # Introduce a scaling factor to replicate HNN++
        self.scale_tau = nn.Parameter(torch.zeros(1))
        self.scale_gamma = nn.Parameter(torch.zeros(1))

        # For logging purposes - save intermediate values within the attention calculation
        self.similarities = torch.nn.ModuleList([IntermediateValueModule(f"similarities.{i}", enabled=self.save_intermediate_values) for i in range(num_heads)])
        self.weights = torch.nn.ModuleList([IntermediateValueModule(f"weights.{i}", enabled=self.save_intermediate_values) for i in range(num_heads)])
        self.aggregates = torch.nn.ModuleList([IntermediateValueModule(f"aggregates.{i}", enabled=self.save_intermediate_values) for i in range(num_heads)])

        if self.use_attention_projection:
            self.attention_projection = HLinear(
                in_features=embed_dim,
                out_features=embed_dim,
                manifold=manifold,
                init_scale=self.mlp_init_scale,
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

        # Apply similarity function f and activation g to Qs and Ks to obtain weights (g(f(q, k)))
        # The code of HNN++ has the following line for similarity calculation, which I tried to replicate:
        # x = - self.ball.dist_matmul(x, encoder_out[0]) / self.scale.exp()
        similarities = []
        for q, k, s in zip(split_q, split_k, self.similarities):
            # Calculate similarity scores
            sim = self.manifold.attention_similarity(queries=q, keys=k, tau=1/self.scale_tau.exp(), gamma=self.scale_gamma)
            
            # Apply causal mask if needed
            if self.causal:
                # Get sequence lengths
                q_len, k_len = q.size(1), k.size(1)
                
                # Create causal mask: lower triangular matrix of ones, upper triangular of zeros
                # This ensures each position can only attend to itself and previous positions
                mask = torch.triu(torch.ones(q_len, k_len, device=sim.device), diagonal=1).bool()
                
                # Apply mask by setting masked positions to -inf (will become 0 after softmax)
                sim = sim.masked_fill(mask, float('-inf'))
            
            similarities.append(s(sim))

        weights = [
            w(self.manifold.attention_activation(s, self.attention_activation)) for s, w in zip(similarities, self.weights)
        ]

        # Aggregate values with sequence-aware specialization of the centroid
        aggregates = [
            a(self.manifold.attention_midpoint(x=v, w=w)) for v, w, a in zip(split_v, weights, self.aggregates)
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

class HMultiheadAttentionOpt(Module):
    """ Multihead attention module for hyperbolic manifolds, without the use of intermediate value modules """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        attention_activation: str = "exp",
        causal: bool = True,
        use_attention_projection: bool = False,
        mlp_init_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self.attention_activation = attention_activation
        self.causal = causal
        self.use_attention_projection = use_attention_projection
        self.mlp_init_scale = mlp_init_scale

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

        # Introduce a scaling factor to replicate HNN++
        self.scale_tau = nn.Parameter(torch.zeros(1))
        self.scale_gamma = nn.Parameter(torch.zeros(1))

        if self.use_attention_projection:
            self.attention_projection = HLinear(
                in_features=embed_dim,
                out_features=embed_dim,
                manifold=manifold,
                init_scale=self.mlp_init_scale,
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

        # Apply similarity function f and activation g to Qs and Ks to obtain weights (g(f(q, k)))
        # The code of HNN++ has the following line for similarity calculation, which I tried to replicate:
        # x = - self.ball.dist_matmul(x, encoder_out[0]) / self.scale.exp()
        similarities = []
        for q, k in zip(split_q, split_k):
            # Calculate similarity scores
            sim = self.manifold.attention_similarity(queries=q, keys=k, tau=1/self.scale_tau.exp(), gamma=self.scale_gamma)
            
            # Apply causal mask if needed
            if self.causal:
                # Get sequence lengths
                q_len, k_len = q.size(1), k.size(1)
                
                # Create causal mask: lower triangular matrix of ones, upper triangular of zeros
                # This ensures each position can only attend to itself and previous positions
                mask = torch.triu(torch.ones(q_len, k_len, device=sim.device), diagonal=1).bool()
                
                # Apply mask by setting masked positions to -inf (will become 0 after softmax)
                sim = sim.masked_fill(mask, float('-inf'))
            
            similarities.append(s(sim))

        weights = [self.manifold.attention_activation(s, self.attention_activation) for s in similarities]

        # Aggregate values with sequence-aware specialization of the centroid
        aggregates = [self.manifold.attention_midpoint(x=v, w=w) for v, w in zip(split_v, weights)]

        # Concatenate outputs
        result = self.manifold.cat(aggregates, dim=-1)

        # Handle non-batch-first output
        if not self.batch_first:
            result = result.transpose(0, 1)

        if need_weights:
            return result, weights
        else:
            return result
