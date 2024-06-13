import jax
import jax.numpy as jnp

from typing import NamedTuple

class RMSNormDescriptor(NamedTuple):
    """
    Contains params for RMS layer normalization
    """
    scale: jnp.ndarray      # (d_model,)


def apply_RMSNorm(RMSDesc: RMSNormDescriptor, x: jnp.ndarray):
    """
    Applies RMS layer normalization to input

    Args:
        RMSDesc: RMS norm params
        x: (B, seq_len, d_model)
    """

    x = x.astype(jnp.float32)
    scale = RMSDesc.scale.astype(jnp.float32)
    normed = x * jnp.reciprocal(jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6))
    return (normed * (1 + scale))


if __name__ == "__main__":
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    scale = jnp.ones(3)*2

    desc = RMSNormDescriptor(scale)

    print(x)
    print("=== Result ===")
    print(apply_RMSNorm(desc, x))

    print("=== Correctness ===")
    expected = jnp.array([0.6324429, 1.2648858])
    x = jnp.array([[0.1, 0.2]])
    desc = RMSNormDescriptor(jnp.zeros(2))
    res = apply_RMSNorm(desc, x)
    print(res.flatten())
    print("Match expected:", jnp.allclose(res.flatten(), expected))
