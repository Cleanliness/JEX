# positional embeddings
import jax
import jax.numpy as jnp

from typing import NamedTuple

class RoPEDescriptor(NamedTuple):
    """
    Contains params for rotary embedding layer

    Assumes embeddings have an even number of dimensions
    i.e. d_model % 2 == 0
    """
    theta: jnp.ndarray  # minimum rotary angles (d_model / 2,)


def apply_RoPE(rotary_descriptor: RoPEDescriptor, x: jnp.ndarray):
    """
    Applies rotary positional encoding to input

    Args:
        rotary_descriptor: RoPEDescriptor
        x: (seq_len, d_model)
    """
    M = x.shape[0]
    theta = rotary_descriptor.theta          # (d_model / 2,)
    m = jnp.arange(0, M)                     # (seq_len,)

    m = jnp.expand_dims(m, axis=1)           # (seq_len, 1)
    angles = m * jnp.repeat(theta, 2)        # (seq_len, d_model)

    # apply rotation block matrix
    rot1 = jnp.cos(angles)                   # (d_model,)
    rot2 = jnp.sin(angles)                   # (d_model,)

    weaved = x.at[::2].set(-x[1::2])
    weaved = weaved.at[1::2].set(x[::2])

    return x*rot1 + weaved*rot2


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10, 4))
    print(x)

    # seq_len = 10, d_model = 4
    theta = jnp.array([0.1, 0.2])
    
    desc = RoPEDescriptor(theta)

    print("=== Result ===")
    
    print(apply_RoPE(desc, x))