import jax
import jax.numpy as jnp

from typing import NamedTuple

class RMSNormDescriptor(NamedTuple):
    """
    Contains params for RMS layer normalization
    """
    scale: jnp.ndarray


def apply_RMSNorm(RMSDesc: RMSNormDescriptor, x: jnp.ndarray):
    """
    Applies RMS layer normalization to input

    Args:
        RMSDesc: RMS norm params
        x: (seq_len, d_model)
    """
    return x / jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True) + 1e-6) * RMSDesc.scale


if __name__ == "__main__":
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    scale = jnp.ones(3)*2

    desc = RMSNormDescriptor(scale)

    print(x)
    print("=== Result ===")
    print(apply_RMSNorm(desc, x))