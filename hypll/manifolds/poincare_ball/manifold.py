import functools
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, empty, eye, no_grad
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import softplus, unfold, softmax
from torch.nn.init import normal_, zeros_

from hypll.manifolds.base import Manifold
from hypll.manifolds.euclidean import Euclidean
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hypll.utils.math import beta_func
from hypll.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_if_man_dims_match,
    check_tangent_tensor_positions,
)

from hypll.manifolds.poincare_ball.math.geooptplus_copied_math_funcs import mobius_scalar_mul

from .math.diffgeom import (
    cdist_mobius,
    cdist_arccosh,
    dist,
    euc_to_tangent,
    expmap,
    expmap0,
    gyration,
    inner,
    logmap,
    logmap0,
    mobius_add,
    project,
    transp,
)
from .math.linalg import poincare_fully_connected, poincare_hyperplane_dists
from .math.stats import attention_midpoint, frechet_mean, frechet_variance, midpoint


class PoincareBall(Manifold):
    """Class representing the Poincare ball model of hyperbolic space.

    Implementation based on the geoopt implementation, but changed to use
    hyperbolic torch functions.

    Attributes:
        c:
            Curvature of the manifold.

    """

    def __init__(self, c: Curvature, cdist_type: str="mobius"):
        """Initializes an instance of PoincareBall manifold.

        Examples:
            >>> from hypll.manifolds.poincare_ball import PoincareBall, Curvature
            >>> curvature = Curvature(value=1.0)
            >>> manifold = Manifold(c=curvature)

        """
        super(PoincareBall, self).__init__()
        self.c = c
        self.cdist_type = cdist_type

    def mobius_add(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(x, y)
        new_tensor = mobius_add(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        new_tensor = project(x=x.tensor, c=self.c(), dim=x.man_dim, eps=eps)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        dim = v.broadcasted_man_dim
        if v.manifold_points is None:
            new_tensor = expmap0(v=v.tensor, c=self.c(), dim=dim)
        else:
            new_tensor = expmap(x=v.manifold_points.tensor, v=v.tensor, c=self.c(), dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def logmap(self, x: Optional[ManifoldTensor], y: ManifoldTensor):
        if x is None:
            dim = y.man_dim
            new_tensor = logmap0(y=y.tensor, c=self.c(), dim=y.man_dim)
        else:
            dim = check_dims_with_broadcasting(x, y)
            new_tensor = logmap(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)
        return TangentTensor(data=new_tensor, manifold_points=x, manifold=self, man_dim=dim)

    def gyration(self, u: ManifoldTensor, v: ManifoldTensor, w: ManifoldTensor) -> ManifoldTensor:
        dim = check_dims_with_broadcasting(u, v, w)
        new_tensor = gyration(u=u.tensor, v=v.tensor, w=w.tensor, c=self.c(), dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def transp(self, v: TangentTensor, y: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(v, y)
        tangent_vectors = transp(
            x=v.manifold_points.tensor, y=y.tensor, v=v.tensor, c=self.c(), dim=dim
        )
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=y,
            manifold=self,
            man_dim=dim,
        )

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return dist(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)

    def inner(
        self, u: TangentTensor, v: TangentTensor, keepdim: bool = False, safe_mode: bool = True
    ) -> Tensor:
        dim = check_dims_with_broadcasting(u, v)
        if safe_mode:
            check_tangent_tensor_positions(u, v)

        return inner(
            x=u.manifold_points.tensor, u=u.tensor, v=v.tensor, c=self.c(), dim=dim, keepdim=keepdim
        )

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(x, u)
        tangent_vectors = euc_to_tangent(x=x.tensor, u=u.tensor, c=self.c(), dim=x.man_dim)
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=x,
            manifold=self,
            man_dim=dim,
        )

    def hyperplane_dists(self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[Tensor]) -> Tensor:
        if x.man_dim != 1 or z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the inputs to be 1 and the manifold "
                f"dimension of the hyperplane orientations to be 0, but got {x.man_dim} and "
                f"{z.man_dim}, respectively"
            )
        return poincare_hyperplane_dists(x=x.tensor, z=z.tensor, r=r, c=self.c())

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        if z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the hyperplane orientations to be 0, but got "
                f"{z.man_dim} instead"
            )
        new_tensor = poincare_fully_connected(
            x=x.tensor, z=z.tensor, bias=bias, c=self.c(), dim=x.man_dim
        )
        new_tensor = ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)
        return self.project(new_tensor)

    def mlr(self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]) -> ManifoldTensor:
        new_tensor = self.hyperplane_dists(x=x, z=z, r=bias)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def frechet_mean(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        output_man_dim = x.man_dim - sum(bd < x.man_dim for bd in batch_dim)
        new_tensor = frechet_mean(
            x=x.tensor, c=self.c(), vec_dim=x.man_dim, batch_dim=batch_dim, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=output_man_dim)

    def midpoint(
        self,
        x: ManifoldTensor,
        batch_dim: int = 0,
        w: Optional[Tensor] = None,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]

        if x.man_dim in batch_dim:
            raise ValueError(
                f"Tried to aggregate over dimensions {batch_dim}, but input has manifold "
                f"dimension {x.man_dim} and cannot aggregate over this dimension"
            )

        # Output manifold dimension is shifted left for each batch dim that disappears
        man_dim_shift = sum(bd < x.man_dim for bd in batch_dim)
        new_man_dim = x.man_dim - man_dim_shift if not keepdim else x.man_dim

        new_tensor = midpoint(
            x=x.tensor, c=self.c(), man_dim=x.man_dim, batch_dim=batch_dim, w=w, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=new_man_dim)
    
    def attention_midpoint(
        self,
        x: ManifoldTensor,
        w: Optional[Tensor] = None,
    ) -> ManifoldTensor:
        if x.dim() != 3:
            raise ValueError(f"Expected inputs to have 3 dimensions, but found {x.dim()}")
        
        if w.dim() != 3:
            raise ValueError(f"Expected weights to have 3 dimensions, but found {w.dim()}")
        
        if x.size(0) != w.size(0) or x.size(1) != w.size(2):
            raise ValueError(
                f"Sizes of inputs and weights indicate either differing batch sizes {x.size(0)}, "
                f"{w.size(0)} or sequence lengths {x.size(1)}, {w.size(2)}"
            )
        
        if x.man_dim != 2:
            raise ValueError(
                f"Expected the manifold dimension of the inputs to be 2, but got {x.man_dim}"
            )
        
        new_tensor = attention_midpoint(
            x=x.tensor, c=self.c(), w=w,
        )
        result = ManifoldTensor(
            data=new_tensor, manifold=self, man_dim=-1
        )
        
        # FIXME HNN++ does a scaling as well, but that's to balance shorter, padded sequences and longer, unpadded sequences
        # if I understand correctly. Probably need to add that.
        # s = w.size(1)  # Fixme check if this is the correct dim
        # result = self.mobius_scalar_mul(torch.FloatTensor([math.sqrt(s)]).to(x.tensor), result)

        # HNN++ does a projection as well, not clear how much difference these make
        # result = self.project(result)
        return result

    def frechet_variance(
        self,
        x: ManifoldTensor,
        mu: Optional[ManifoldTensor] = None,
        batch_dim: Union[int, list[int]] = -1,
        keepdim: bool = False,
    ) -> Tensor:
        if mu is not None:
            mu = mu.tensor

        # TODO: Check if x and mu have compatible man_dims
        return frechet_variance(
            x=x.tensor,
            c=self.c(),
            mu=mu,
            vec_dim=x.man_dim,
            batch_dim=batch_dim,
            keepdim=keepdim,
        )

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=Euclidean(),
            man_dim=0,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Optional[Parameter]) -> None:
        in_features, out_features = weight.size()
        if in_features <= out_features:
            with no_grad():
                weight.tensor.copy_(1 / 2 * eye(in_features, out_features))
        else:
            normal_(
                weight.tensor,
                mean=0,
                std=(2 * in_features * out_features) ** -0.5,
            )
        if bias is not None:
            zeros_(bias)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> ManifoldTensor:
        # TODO: may have to cache some of this stuff for efficiency.
        in_channels = input.size(1)
        if len(kernel_size) == 2:
            kernel_vol = kernel_size[0] * kernel_size[1]
        else:
            kernel_vol = kernel_size**2
            kernel_size = (kernel_size, kernel_size)

        beta_ni = beta_func(in_channels / 2, 1 / 2)
        beta_n = beta_func(in_channels * kernel_vol / 2, 1 / 2)

        input = self.logmap(x=None, y=input)
        input.tensor = input.tensor * beta_n / beta_ni
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        new_tensor = TangentTensor(data=new_tensor, manifold_points=None, manifold=self, man_dim=1)
        return self.expmap(new_tensor)

    def flatten(self, x: ManifoldTensor, start_dim: int = 1, end_dim: int = -1) -> ManifoldTensor:
        """Flattens a manifold tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        If the manifold dimension of the input tensor is among the dimensions which
        are flattened, applies beta-concatenation to the points on the manifold.
        Otherwise simply flattens the tensor using torch.flatten.

        Updates the manifold dimension if necessary.

        """
        start_dim = x.dim() + start_dim if start_dim < 0 else start_dim
        end_dim = x.dim() + end_dim if end_dim < 0 else end_dim

        # Get the range of dimensions to flatten.
        dimensions_to_flatten = x.shape[start_dim + 1 : end_dim + 1]

        if start_dim <= x.man_dim and end_dim >= x.man_dim:
            # Use beta concatenation to flatten the manifold dimension of the tensor.
            #
            # Start by applying logmap at the origin and computing the betas.
            tangents = self.logmap(None, x)
            n_i = x.shape[x.man_dim]
            n = n_i * functools.reduce(lambda a, b: a * b, dimensions_to_flatten)
            beta_n = beta_func(n / 2, 0.5)
            beta_n_i = beta_func(n_i / 2, 0.5)
            # Flatten the tensor and rescale.
            tangents.tensor = torch.flatten(
                input=tangents.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            tangents.tensor = tangents.tensor * beta_n / beta_n_i
            # Set the new manifold dimension
            tangents.man_dim = start_dim
            # Apply exponential map at the origin.
            return self.expmap(tangents)
        else:
            flattened = torch.flatten(
                input=x.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            man_dim = x.man_dim if end_dim > x.man_dim else x.man_dim - len(dimensions_to_flatten)
            return ManifoldTensor(data=flattened, manifold=x.manifold, man_dim=man_dim)

    def cdist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        if self.cdist_type == "mobius":
            return cdist_mobius(x=x.tensor, y=y.tensor, c=self.c())
        elif self.cdist_type == "arccosh":
            return cdist_arccosh(x=x.tensor, y=y.tensor, c=self.c())
        else:
            raise ValueError(f"Unknown cdist type: {self.cdist_type}")

    def cat(
        self,
        manifold_tensors: Union[Tuple[ManifoldTensor, ...], List[ManifoldTensor]],
        dim: int = 0,
    ) -> ManifoldTensor:
        check_if_man_dims_match(manifold_tensors)
        if dim == manifold_tensors[0].man_dim:
            tangent_tensors = [self.logmap(None, t) for t in manifold_tensors]
            ns = torch.tensor([t.shape[t.man_dim] for t in manifold_tensors])
            n = ns.sum()
            beta_ns = beta_func(ns / 2, 0.5)
            beta_n = beta_func(n / 2, 0.5)
            cat = torch.cat(
                [(t.tensor * beta_n) / beta_n_i for (t, beta_n_i) in zip(tangent_tensors, beta_ns)],
                dim=dim,
            )
            new_tensor = TangentTensor(data=cat, manifold=self, man_dim=dim)
            return self.expmap(new_tensor)
        else:
            cat = torch.cat([t.tensor for t in manifold_tensors], dim=dim)
            man_dim = manifold_tensors[0].man_dim
            return ManifoldTensor(data=cat, manifold=self, man_dim=man_dim)

    def split(
        self,
        manifold_tensor: ManifoldTensor,
        split_size_or_sections: Union[int, list[int]],
        dim: int = 0,
    ) -> list[ManifoldTensor]:
        # If we don't split along the man_dim we can use PyTorch's split
        if dim != manifold_tensor.man_dim:
            split_tensors = torch.split(
                tensor=manifold_tensor.tensor,
                split_size_or_sections=split_size_or_sections,
                dim=dim,
            )
            return [
                ManifoldTensor(
                    data=t,
                    manifold=self,
                    man_dim=manifold_tensor.man_dim
                ) for t in split_tensors
            ]
        
        man_dim_size = manifold_tensor.size(dim=dim)

        # Replace the split_size_or_sections by a list if an int is given
        if isinstance(split_size_or_sections, int):
            ssos = (man_dim_size // split_size_or_sections) * [split_size_or_sections]
            remainder = man_dim_size % split_size_or_sections
            if remainder:
                ssos.append(remainder)
        else:
            ssos = split_size_or_sections

        # Map to tangent space
        v = self.logmap(x=None, y=manifold_tensor)

        # Split and scale
        v_split_tensors = v.tensor.split(split_size=ssos, dim=dim)
        beta_n = beta_func(man_dim_size / 2, 0.5)
        beta_ni = [beta_func(ni / 2, 0.5) for ni in ssos]
        scaled_v_split_tensors = [bni / beta_n * vt for bni, vt in zip(beta_ni, v_split_tensors)]

        # Map back to manifold and return
        new_tensors = [
            TangentTensor(data=svt, manifold=self, man_dim=dim) for svt in scaled_v_split_tensors
        ]
        return [
            self.expmap(v=t) for t in new_tensors
        ]

    # TODO: Find good default values for tau and gamma -> aren't mentioned in HNN++ paper
    def attention_similarity(
        self, queries: ManifoldTensor, keys: ManifoldTensor, tau: float = 10.0, gamma: float = 1.0
    ) -> Tensor:
        return -tau * self.cdist(queries, keys) - gamma

    def attention_activation(self, similarities: Tensor) -> Tensor:
        return similarities.exp()
        # return softmax(similarities, dim=-1)  # other option from HNN++, not clear if it's better

    def mobius_scalar_mul(self, r: torch.Tensor, x: ManifoldTensor, *, dim=-1) -> ManifoldTensor:
        """ Added from HNN++, but uses geoopt functions under the hood
        FIXME integrate properly if we keep on using it """
        dim = check_dims_with_broadcasting(r, x)
        # to the best of my understanding, c == k, but I wasn't entirely sure
        new_tensor = mobius_scalar_mul(r=r, x=x.tensor, k=self.c(), dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

