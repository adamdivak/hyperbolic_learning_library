from torch.nn import Module, LayerNorm

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


# PyTorch def
from torch.nn.modules.normalization import _shape_t

class HLayerNorm(Module):
    """ LayerNorm - inefficient implementation in tangent space.
    Signature identical to PyTorch's LayerNorm, except for the additional manifold parameter """
    def __init__(self, normalized_shape: _shape_t,
                manifold: Manifold,
                eps: float = 1e-5,
                elementwise_affine: bool = True,
                bias: bool = True,
                device=None,
                dtype=None,
        ) -> None:
        
        super(HLayerNorm, self).__init__()
        self.manifold = manifold
        self.euclidean_ln = LayerNorm(normalized_shape, 
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=self.euclidean_ln,
            manifold=self.manifold,
            input=input,
        )
