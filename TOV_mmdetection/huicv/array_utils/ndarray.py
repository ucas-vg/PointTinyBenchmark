import torch


def inverse_dim(permute_dim):
    """
    Examples:
        from random import shuffle
        a = torch.randn(1, 2, 3, 4, 5)
        permute_dim = list(range(len(a.shape)))
        shuffle(permute_dim)
        b = a.permute(permute_dim)
        b = b.permute(inverse_dim(permute_dim))
        assert (a == b).all()
    Args:
        permute_dim: (tuple, list)
    Returns:
    """
    # (0, 1, 2, 3) => (0, 3, 1, 2); dim_map[0]=0, dim_map[1]=2, dim_map[2]=3, dim_map[3]=1
    # (idx[0], idx[1], idx[2], idx[3]) = (0, 2, 3, 1)
    dim_map = [-1] * len(permute_dim)
    for i, d in enumerate(permute_dim):  # new_dim, old_dim
        dim_map[d] = i
    return dim_map


def take_last_dim(x, idx):
    """
    Examples:
        a = torch.randn(3, 4, 5, 2)
        ta, idx = a.sum(dim=(-1, -2)).topk(3, dim=-1)  # (3, 3)
        assert (ta == take_last_dim(a, idx).sum(dim=(-1, -2))).all()
    Args:
        x: shape=(s1, s2, ...s(n-1), sn, ...)
        idx: shape=(s1, s2, ...s(n-1), m)
    Returns:
    """
    shape1, m = idx.shape[:-1], idx.shape[-1]  # (s1, s2, ..s(n-1)), m
    x_shape1, n, shape2 = x.shape[:len(shape1)], x.shape[len(shape1)], x.shape[len(shape1)+1:]
    x = x.reshape(-1, n, *shape2)  # (s1*s2...*s(n-1), n, ...)
    idx = idx.reshape(-1, m)         # (s1*s2...*s(n-1), m)
    assert x_shape1 == shape1

    for_idx = torch.arange(len(x)).reshape((-1, 1)).repeat((1, m)).to(idx.device)  # (s1*s2...*s(n-1), m)
    x = x[for_idx, idx].reshape(*shape1, m, *shape2)
    return x


def take(x, idx, dim, pass_dim=()):
    """
    dim=2, idx_dim=1, pass_dim=(1,)
    x.shape:   (2, 3, 6, 5) => (2, 5, 6, 3) => (2, 5, 4, 3) => (2, 3, 4, 5)
    idx.shape: (2,    4, 5) => (2, 5, 4)                    => (2,    4, 5)
    1. get permute_dim of x => (0, 1, 2, 3)
    2. remove pass dim(1,) => (0, 2, 3) which match shape of idx
    3. find index of dim in (0, 2, 3) => matched idx_dim = 1
    4. remove dim(2) => (0, 3)
    5. append dim and pass_dim => (0, 3, 2, 1)

    Example:
        a = torch.randn(3, 4, 5, 2)
        ta, idx = a.sum(dim=1).topk(3, dim=1)  # (3, 3, 2)
        assert (ta == take(a, idx, dim=2, pass_dim=1).sum(dim=1)).all()
    Args:
        x: shape=(s1, s2, ..sk, s(k+1), s(k+2),... sn) or (s1, s2, ..p1, ..p2, .sk, s(k+1), s(k+2),..p(t+1),.. sn),
            where p* in the dim of pass_dim.
        idx: shape=(s1, s2, ..sk, m, s(k+2), ...sn)
    Returns:
    """
    if isinstance(pass_dim, int):
        pass_dim = (pass_dim,)

    dim = dim % len(x.shape)
    pass_dim = tuple(d % len(x.shape) for d in pass_dim)

    if len(pass_dim) == 0:
        first_dim_x = list(x.shape[:dim]) + [":"] + list(x.shape[dim+1:len(idx.shape)])
        first_dim_idx = list(idx.shape[:dim]) + [":"] + list(idx.shape[dim+1:])
        assert first_dim_x == first_dim_idx, \
            f"first {len(idx.shape)} dim(except dim{dim}) of x must match with idx," \
            f" but got {first_dim_x} vs {first_dim_idx}. if you want to keep all element in the unmatched dim, " \
            f"set them in pass_dim."
        pass_dim = range(len(idx.shape), len(x.shape))

    permute_dims = list(range(len(x.shape)))
    for d in pass_dim:
        permute_dims.remove(d)
    idx_dim = permute_dims.index(dim)
    permute_dims.remove(dim)
    permute_dims.append(dim)
    permute_dims.extend(pass_dim)

    idx_permute_dims = list(range(len(idx.shape)))
    idx_permute_dims.remove(idx_dim)
    idx_permute_dims.append(idx_dim)

    x = x.permute(permute_dims)
    idx = idx.permute(idx_permute_dims)
    x = take_last_dim(x, idx)
    x = x.permute(inverse_dim(permute_dims))
    return x


if __name__ == '__main__':
    from random import shuffle
    a = torch.randn(1, 2, 3, 4, 5)
    permute_dim = list(range(len(a.shape)))
    shuffle(permute_dim)
    b = a.permute(permute_dim)
    b = b.permute(inverse_dim(permute_dim))
    assert (a == b).all()

    a = torch.randn(100, 4, 5, 2)
    ta, idx = a.sum(dim=(-1, -2)).topk(3, dim=1)  # (100, 3)
    assert (ta == take_last_dim(a, idx).sum(dim=(-1, -2))).all()
    assert (ta == take(a, idx, dim=1).sum(dim=(-1, -2))).all()

    a = torch.randn(100, 4, 5, 2)
    ta, idx = a.sum(dim=1).topk(3, dim=1)  # (100, 3, 2)
    assert (ta == take(a, idx, dim=2, pass_dim=1).sum(dim=1)).all()

    a = torch.randn(100, 4, 5, 2)
    ta, idx = a.sort(dim=1)
    assert (ta[:, :2] == take(a, idx[:, :2], dim=1)).all()

