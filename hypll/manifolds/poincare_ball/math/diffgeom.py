import torch


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    broadcasted_dim = max(x.dim(), y.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    x2 = x.pow(2).sum(
        dim=dim - broadcasted_dim + x.dim(),
        keepdim=True,
    )
    y2 = y.pow(2).sum(
        dim=dim - broadcasted_dim + y.dim(),
        keepdim=True,
    )
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / denom.clamp_min(1e-15)


def project(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5)
    maxnorm = torch.where(c.gt(0), maxnorm, c.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def expmap0(v: torch.Tensor, c: torch.Tensor, dim: int = -1):
    v_norm_c_sqrt = v.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return project(torch.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c, dim=dim)


def logmap0(y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    y_norm_c_sqrt = y.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt


def expmap(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = -1):
    broadcasted_dim = max(x.dim(), v.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    v_norm = v.norm(dim=dim - broadcasted_dim + v.dim(), keepdim=True).clamp_min(1e-15)
    lambda_x = 2 / (
        1 - c * x.pow(2).sum(dim=dim - broadcasted_dim + x.dim(), keepdim=True)
    ).clamp_min(1e-15)
    c_sqrt = c.sqrt()
    second_term = torch.tanh(c_sqrt * lambda_x * v_norm / 2) * v / (c_sqrt * v_norm)
    return project(mobius_add(x, second_term, c, dim=dim), c, dim=dim)


def logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    broadcasted_dim = max(x.dim(), y.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    min_x_y = mobius_add(-x, y, c, dim=dim)
    min_x_y_norm = min_x_y.norm(dim=dim, keepdim=True).clamp_min(1e-15)
    lambda_x = 2 / (
        1 - c * x.pow(2).sum(dim=dim - broadcasted_dim + x.dim(), keepdim=True)
    ).clamp_min(1e-15)
    c_sqrt = c.sqrt()
    return 2 / (c_sqrt * lambda_x) * torch.atanh(c_sqrt * min_x_y_norm) * min_x_y / min_x_y_norm


def gyration(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
):
    broadcasted_dim = max(u.dim(), v.dim(), w.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    u2 = u.pow(2).sum(dim=dim - broadcasted_dim + u.dim(), keepdim=True)
    v2 = v.pow(2).sum(dim=dim - broadcasted_dim + v.dim(), keepdim=True)
    uv = (u * v).sum(dim=dim - broadcasted_dim + max(u.dim(), v.dim()), keepdim=True)
    uw = (u * w).sum(dim=dim - broadcasted_dim + max(u.dim(), w.dim()), keepdim=True)
    vw = (v * w).sum(dim=dim - broadcasted_dim + max(v.dim(), w.dim()), keepdim=True)
    K2 = c**2
    a = -K2 * uw * v2 + c * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)


def transp(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
):
    broadcasted_dim = max(x.dim(), y.dim(), v.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    lambda_x = 2 / (
        1 - c * x.pow(2).sum(dim=dim - broadcasted_dim + x.dim(), keepdim=True)
    ).clamp_min(1e-15)
    lambda_y = 2 / (
        1 - c * y.pow(2).sum(dim=dim - broadcasted_dim + y.dim(), keepdim=True)
    ).clamp_min(1e-15)
    return gyration(y, -x, v, c, dim=dim) * lambda_x / lambda_y


def dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    return (
        2
        / c.sqrt()
        * (c.sqrt() * mobius_add(-x, y, c, dim=dim).norm(dim=dim, keepdim=keepdim)).atanh()
    )


def inner(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    broadcasted_dim = max(x.dim(), u.dim(), v.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    lambda_x = 2 / (
        1 - c * x.pow(2).sum(dim=dim - broadcasted_dim + x.dim(), keepdim=True)
    ).clamp_min(1e-15)
    dot_prod = (u * v).sum(dim=dim, keepdim=keepdim)
    return lambda_x.square() * dot_prod


def euc_to_tangent(
    x: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    broadcasted_dim = max(x.dim(), u.dim())
    dim = dim if dim >= 0 else broadcasted_dim + dim
    lambda_x = 2 / (
        1 - c * x.pow(2).sum(dim=dim - broadcasted_dim + x.dim(), keepdim=True)
    ).clamp_min(1e-15)
    return u / lambda_x**2


def cdist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return cdist_mobius(x, y, c)
    # return cdist_arccosh(x, y, c)

def cdist_arccosh(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    # Handle batched inputs with shape [batch, num_points, dim]
    # and compute pairwise distances between all points in x and y
    # This is an alternative, equivalent implementation of cdist_mobius, which 
    # is supposed to be more numerically stable
    
    # Expand dimensions for broadcasting
    # x shape: [batch, n_x, dim] -> [batch, n_x, 1, dim]
    # y shape: [batch, n_y, dim] -> [batch, 1, n_y, dim]
    x_e = x.unsqueeze(2)
    y_e = y.unsqueeze(1)
    
    # Compute squared difference: ||x - y||^2
    diff_sq = (x_e - y_e).pow(2).sum(dim=-1)  # [batch, n_x, n_y]
    
    # Compute ||x||^2 and ||y||^2
    x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)  # [batch, n_x, 1]
    y_norm_sq = y.pow(2).sum(dim=-1, keepdim=True)  # [batch, n_y, 1]
    
    # Compute denominator: (1 - c||x||^2)(1 - c||y||^2)
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq.transpose(1, 2))  # [batch, n_x, n_y]
    
    # Compute the full formula: 1 / sqrt(c) * arccosh(1 + 2*c*num/denom)
    inner = 1 + 2 * c * diff_sq / denom.clamp_min(1e-15)
    inner_clamped = inner.clamp(min=1.0 + 1e-15) # had to introduce clamping here to avoid nans
    result = 1 / c.sqrt() * inner_clamped.arccosh()
    assert not inner.isnan().any(), inner
    assert not inner.isinf().any(), inner
    assert not result.isnan().any(), result
    assert not result.isinf().any(), result
    return result


def cdist_mobius(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    inner = c.sqrt() * mobius_add_batch(-x, y, c).norm(dim=-1)
    eps = 1e-7 
    inner_clamped = inner.clamp(min=-1.0 + eps, max=1.0 - eps)
    inner_tanh = inner_clamped.atanh() # had to introduce clamping here to avoid nans
    result = 2 / c.sqrt() * inner_tanh
    # FIXME re-introduce
    # assert not result.isnan().any(), result
    # assert not result.isinf().any(), result
    return result


def mobius_add_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    xy = torch.einsum("bij,bkj->bik", (x, y))
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    num = 1 + 2 * c * xy + c * y2.permute(0, 2, 1)
    num = num.unsqueeze(3) * x.unsqueeze(2)
    num = num + (1 - c * x2).unsqueeze(3) * y.unsqueeze(1)
    denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(0, 2, 1)
    return num / denom.unsqueeze(3).clamp_min(1e-15)
