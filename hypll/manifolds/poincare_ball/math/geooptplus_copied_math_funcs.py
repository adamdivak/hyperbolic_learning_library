# Copied from geooptplus of the HNN++ repo
# Copied all functions that are needed for mobius scalar multiplication

from typing import List, Optional, Union, Tuple
import math
import torch
from torch import nn
from torch.nn import Module

def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
            - 1 / 11 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x ** 3
    elif order == 2:
        return x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5 - 1 / 7 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

def tanh(x):
    return x.clamp(-15, 15).tanh()


def sabs(x, eps: float = 1e-15):
    #return x.abs().add_(eps)
    return x.abs().clamp_min(eps)

def sign(x):
    return torch.sign(x.sign() + 0.5)

def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)

def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
            + 1382 / 155925 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x ** 3
    elif order == 2:
        return x + 1 / 3 * k * x ** 3 + 2 / 15 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)


def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)

def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius scalar multiplication.

    .. math::

        r \otimes_\kappa x
        =
        \tan_\kappa(r\tan_\kappa^{-1}(\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_\kappa x = x \oplus_\kappa \dots \oplus_\kappa x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_\kappa x
         =
         r_1 \otimes_\kappa x \oplus r_2 \otimes_\kappa x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_\kappa x = r_1 \otimes_\kappa (r_2 \otimes_\kappa x)

    * Monodistributivity

    .. math::

         r \otimes_\kappa (r_1 \otimes x \oplus r_2 \otimes x) =
         r \otimes_\kappa (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_\kappa x / \|r \otimes_\kappa x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : tensor
        scalar for multiplication
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, k, dim=dim)


@torch.jit.script
def _mobius_scalar_mul(
    r: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(r * artan_k(x_norm, k), k) * (x / x_norm)
    return res_c
