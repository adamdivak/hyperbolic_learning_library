# %%
import numpy as np
import torch

import sys
sys.path.append("/Users/adamdivak/Documents/Study/Thesis/hyperbolic_learning_library")
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.nn.modules.linear import HLinear

torch.manual_seed(42)

# %%
manifold = Euclidean()

# %%
inp = torch.randn(3, 4)

# %%
hypll_euc_linear = HLinear(
    in_features=4,
    out_features=4,
    manifold=manifold,
    bias=True
)

# %%
# For the Euclidean case it doesn't matter if we directly materialize the ManifoldTensor,
# or if we map from the TangentTensor
inp_m = ManifoldTensor(data=inp, manifold=manifold)
# inp_t = TangentTensor(data=inp, man_dim=-1, manifold=manifold)
# inp_m = manifold.expmap(inp_t)
hypll_euc_linear_out = hypll_euc_linear(inp_m).detach()
print(hypll_euc_linear_out.tensor.shape)
print(hypll_euc_linear_out.tensor)

# %%

torch_linear_out = torch.nn.functional.linear(
    input=inp, 
    weight=hypll_euc_linear.z.tensor, 
    bias=hypll_euc_linear.bias
).detach()
print(torch_linear_out.shape)
print(torch_linear_out)

# %%
assert np.allclose(hypll_euc_linear_out.tensor, torch_linear_out)
