from torch.nn import Module, LayerNorm
import math

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


# PyTorch def
from torch.nn.modules.normalization import _shape_t

class HLayerNorm(Module):
    """ LayerNorm - inefficient implementation in tangent space.
    Signature identical to PyTorch's LayerNorm, except for the additional manifold parameter, plus a division by sqrt(dim) """
    def __init__(self, normalized_shape: _shape_t,
                manifold: Manifold,
                eps: float = 1e-5,
                elementwise_affine: bool = True,
                bias: bool = True,
                scale_by_d: bool = False,
                device=None,
                dtype=None,
        ) -> None:
        
        super(HLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.manifold = manifold
        self.euclidean_ln = LayerNorm(normalized_shape, 
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            scale_by_d=scale_by_d,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            # Original LayerNorm calculation divided by sqrt(dim), in order to keep the norm
            # of the points identical.
            # FIXME: math.sqrt(self.normalized_shape) currently only works for int shapes, handle multidim shapes here
            op=lambda x: self.euclidean_ln(x) / math.sqrt(self.normalized_shape) if not self.scale_by_d else self.euclidean_ln(x),
            manifold=self.manifold,
            input=input,
        )
