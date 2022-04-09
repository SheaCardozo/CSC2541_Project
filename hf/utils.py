import jax
from jax import jit, tree_map
from jax import numpy as np
from jax.tree_util import tree_reduce


@jit
def dot(u, v):
    dots = tree_map(lambda x, y: np.vdot(x, y), u, v)
    return jax.tree_util.tree_reduce(lambda acc, node: acc + node, dots, 0)


@jit
def norm(v):
    return np.sqrt(dot(v, v))


@jit
def zero_vec(v):
    return tree_map(lambda x: np.zeros_like(x), v)


@jit
def one_vec(v):
    return tree_map(lambda x: np.ones_like(x), v)


# @jit
# def copy_vec(v):
#     return tree_map(lambda x: np.array(x), v)


@jit
def dot(u, v):
    dots = tree_map(lambda x, y: np.vdot(x, y), u, v)
    return tree_reduce(
        lambda acc, node: acc + node, dots, 0)


@jit
def lin_comb(u, k, v):
    return tree_map(lambda x, y: x + k * y, u, v)


@jit
def hadamard(u, v):
    return tree_map(lambda x, y: x * y, u, v)


@jit
def scale_vec(k, v):
    return tree_map(lambda x: k * x, v)


@jit
def normalize(x):
    return np.nan_to_num(x / np.linalg.norm(x))
