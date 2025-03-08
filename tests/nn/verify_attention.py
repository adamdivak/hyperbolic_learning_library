# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/adamdivak/Documents/Study/Thesis/hyperbolic_learning_library")
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.nn.modules.attention import HMultiheadAttention

# %%
manifold = Euclidean()

# %%
torch.manual_seed(42)  # For reproducibility

# Inputs
q = torch.randn(1, 3, 4)  # B, S, D
k = torch.randn(1, 3, 4)
v = torch.randn(1, 3, 4)

# %%
# Set up HypLL MHA
hypll_euc_attn = HMultiheadAttention(
    embed_dim=4,
    num_heads=1,  # Use a single head for now as the simplest case
    manifold=manifold,
    kdim=None,
    vdim=None,
    batch_first=True # FIXME this had no effect originally, but comments suggests True is the current setup
)

hypll_euc_attn.q_map.bias.tensor = torch.zeros(4)
hypll_euc_attn.k_map.bias.tensor = torch.zeros(4)
hypll_euc_attn.v_map.bias.tensor = torch.zeros(4)

# %%
# Use identity matrix for output projection to leave embeddings unchanged after concatenation,
# As HypLL does not currently have an output projection layer
out_weight = torch.eye(4)

# Use PyTorch's implementation with the same weights
# Note: PyTorch expects weights in a specific format that matches its internal dimension ordering
# Since we're using batch-first in our custom implementation but PyTorch uses batch-middle,
# we need to ensure the weights are compatible

# Transpose the weights for PyTorch
# PyTorch expects weights in the format [out_features, in_features]
# while our custom implementation might have them as [in_features, out_features]
q_weight_transposed = hypll_euc_attn.q_map.z.tensor.transpose(0, 1)
k_weight_transposed = hypll_euc_attn.k_map.z.tensor.transpose(0, 1)
v_weight_transposed = hypll_euc_attn.v_map.z.tensor.transpose(0, 1)

# Use PyTorch's implementation with the transposed weights
attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
    query=q.transpose(1, 0),  # input is batch-first, but torch expects batch-middle
    key=k.transpose(1, 0),
    value=v.transpose(1, 0),
    embed_dim_to_check=4,
    num_heads=1,
    in_proj_weight=None,
    in_proj_bias=None,
    bias_k=None,
    bias_v=None,
    add_zero_attn=False,
    dropout_p=0.0,
    out_proj_weight=out_weight,
    out_proj_bias=None,
    need_weights=True,
    use_separate_proj_weight=True,
    # Use the transposed weights to match PyTorch's expected format
    q_proj_weight=q_weight_transposed,
    k_proj_weight=k_weight_transposed,
    v_proj_weight=v_weight_transposed,
    average_attn_weights=True,
    is_causal=False,
)
attn_output = attn_output.detach().transpose(1, 0)  # Transpose back the output to batch-first
attn_output_weights = attn_output_weights.detach()
print("PyTorch Attention:")
print(attn_output.shape)
print(attn_output)

print(f"\nPyTorch Weights, sum={attn_output_weights.sum():.1f}")
print(attn_output_weights)

# %%
# For the Euclidean case it doesn't matter if we directly materialize the ManifoldTensor,
# or if we map from the TangentTensor
q_m = ManifoldTensor(data=q, manifold=manifold)
k_m = ManifoldTensor(data=k, manifold=manifold)
v_m = ManifoldTensor(data=v, manifold=manifold)

# %%

# Run the full implementation
hypll_euc_attn_output, hypll_euc_attn_weights = hypll_euc_attn(q_m, k_m, v_m, return_weights=True)
hypll_euc_attn_output = hypll_euc_attn_output.tensor.detach()
hypll_euc_attn_weights = [w.detach() for w in hypll_euc_attn_weights]
print("HypLL attention:")
print(hypll_euc_attn_output.shape)
print(hypll_euc_attn_output)

print(f"\nHypLL Weights, sum={sum(w.sum() for w in hypll_euc_attn_weights):.1f}")
print(f"{len(hypll_euc_attn_weights)} * {hypll_euc_attn_weights[0].shape}")
print(hypll_euc_attn_weights)

# %%
# Compare the outputs and visualize the differences
print("\nComparing outputs:")
print(f"PyTorch output shape: {attn_output.shape}")
print(f"Custom output shape: {hypll_euc_attn_output.shape}")

# Calculate difference without normalization
diff_attn = np.abs(attn_output - hypll_euc_attn_output)
diff_weights = np.abs(attn_output_weights - hypll_euc_attn_weights[0])
print(f"Attention max absolute difference: {diff_attn.max():.2f}")
print(f"Attention mean absolute difference: {diff_attn.mean():.2f}")
print(f"Weights max absolute difference: {diff_weights.max():.2f}")
print(f"Weights mean absolute difference: {diff_weights.mean():.2f}")

# Visualize the outputs and their difference
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))

# For raw outputs
vmin_attn = min(attn_output.min().item(), hypll_euc_attn_output.min().item())
vmax_attn = max(attn_output.max().item(), hypll_euc_attn_output.max().item())

# Display raw outputs
im1 = ax1.imshow(attn_output[0], vmin=vmin_attn, vmax=vmax_attn)
im2 = ax2.imshow(hypll_euc_attn_output[0], vmin=vmin_attn, vmax=vmax_attn)
im3 = ax3.imshow(diff_attn[0], cmap='hot', vmin=vmin_attn, vmax=vmax_attn)

vmin_weights = min(attn_output_weights[0].min().item(), hypll_euc_attn_weights[0].min().item())
vmax_weights = max(attn_output_weights[0].max().item(), hypll_euc_attn_weights[0].max().item())

# Display weights
im4 = ax4.imshow(attn_output_weights[0], cmap='hot', vmin=vmin_weights, vmax=vmax_weights)
im5 = ax5.imshow(hypll_euc_attn_weights[0][0], cmap='hot', vmin=vmin_weights, vmax=vmax_weights)
im6 = ax6.imshow(diff_weights[0], cmap='hot', vmin=vmin_weights, vmax=vmax_weights)

title_suffix = '(Raw)'

fig.subplots_adjust(right=0.9)
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
fig.colorbar(im1, cax=cbar_ax1)
fig.colorbar(im3, cax=cbar_ax2)

ax1.set_title(f'PyTorch Attention {title_suffix}')
ax2.set_title(f'HypLL+Euclidean manifold Attention {title_suffix}')
ax3.set_title(f'Absolute Difference Attention {title_suffix}')
ax4.set_title(f'PyTorch Attention Weights')
ax5.set_title(f'HypLL+Euclidean manifold Attention Weights')
ax6.set_title(f'Absolute Difference Weights')

fig.suptitle('Comparison of MultiHeadAttention implementations for the same random input matrices (1 head, 4 embed dims)')

plt.savefig('euclidean_attention_comparison.png')
plt.show()

# %%
import numpy as np
assert np.allclose(attn_output, hypll_euc_attn_output)
