from torch.nn import Module, Dropout

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


class HDropout(Module):
    """ Dropout - inefficient implementation in tangent space.
    Signature identical to PyTorch's Dropout, except for the additional manifold parameter and no inplace option """
    def __init__(self, 
                p: float,
                manifold: Manifold,
        ) -> None:
        
        super().__init__()
        self.manifold = manifold
        self.euclidean_dropout = Dropout(p=p)

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        # Avoid unnecessary tangent space mapping if dropout is disabled
        if self.euclidean_dropout.p > 0.0:
            return op_in_tangent_space(
                op=lambda x: self.euclidean_dropout(x),
                manifold=self.manifold,
                input=input,
            )
        else:
            return input
