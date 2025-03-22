from torch.nn import Module
from torch.nn.functional import relu, leaky_relu, gelu

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


class HReLU(Module):
    def __init__(self, manifold: Manifold) -> None:
        super(HReLU, self).__init__()
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=relu,
            manifold=self.manifold,
            input=input,
        )

# Custom hyperbolic leaky ReLU to prevent dead neurons
class HLeakyReLU(Module):
    def __init__(self, manifold, negative_slope=0.1):
        super().__init__()
        self.manifold = manifold
        self.negative_slope = negative_slope
        
    def forward(self, input):
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=lambda x: leaky_relu(x, negative_slope=self.negative_slope),
            manifold=self.manifold,
            input=input,
        )

class HGELU(Module):
    def __init__(self, manifold: Manifold) -> None:
        super(HGELU, self).__init__()
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=gelu,
            manifold=self.manifold,
            input=input,
        )