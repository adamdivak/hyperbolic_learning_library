from torch.nn import Module, Parameter
from torch.nn.functional import relu, leaky_relu, gelu
from torch import zeros

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space
from scipy.special import beta


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

class HGLU(Module):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.scale = Parameter(zeros(1))

    def forward(self, x, dim=None):
        # If dim is not specified, use the manifold dimension
        if dim is None:
            dim = x.man_dim
        
        channels = x.size(dim)
        
        # Split the input tensor into two equal parts along the specified dimension
        # This uses the manifold's split operation which handles the beta scaling internally
        xa_tensor, xb_tensor = self.manifold.split(x, split_size_or_sections=[channels // 2, channels // 2], dim=dim)
        
        # Map xb to tangent space for applying sigmoid
        xb_tangent = self.manifold.logmap(x=None, y=xb_tensor)
        
        # Apply sigmoid activation in tangent space with scaling
        sigmoid_input = xb_tangent.tensor * (channels ** 0.5) * self.scale.exp()
        activated = sigmoid_input.sigmoid()
        
        # Map xa to tangent space
        xa_tangent = self.manifold.logmap(x=None, y=xa_tensor)
        
        # Element-wise multiplication in tangent space
        result_tangent = xa_tangent.tensor * activated
        
        # Create a proper TangentTensor
        result_tangent_tensor = TangentTensor(data=result_tangent, manifold=self.manifold, man_dim=dim)
        
        # Map back to the manifold
        return self.manifold.expmap(result_tangent_tensor)
