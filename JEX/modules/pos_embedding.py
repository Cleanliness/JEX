# positional embeddings
import jax
import jax.numpy as jnp

from typing import NamedTuple, Optional

# pytorch impl of RoPE
# https://nn.labml.ai/transformers/rope/index.html


class RoPEDescriptor(NamedTuple):
    """
    Contains params for rotary embedding layer

    Assumes an even number of dimensions
    """
    theta: Optional[jnp.array]   # minimum rotary angles (H_q / 2,)
    theta_max: int               # max rotation period for subspaces (see paper)


def init_RoPE(d: int, theta_max: int=10000):
    """
    Initialize rotary positional encoding

    Args:
        theta_max: max rotation period for subspaces
        d: embedding dimension
    """
    assert d % 2 == 0, "embedding dimension must be even"

    theta_in = 2*jnp.arange(0, d//2) / d
    theta = theta_max**theta_in

    return RoPEDescriptor(theta, theta_max)


def apply_RoPE(x: jnp.ndarray, m: jnp.ndarray, rotary_descriptor: RoPEDescriptor):
    """
    Applies rotary positional encoding to input.
    Position indices may not necessarily start from 0,
    since we might be constructing a KV cache.

    Args:
        rotary_descriptor: RoPEDescriptor
        x: (seq_len, d) input, bfloat16
        m: (seq_len,) position indices
    """
    theta = rotary_descriptor.theta          # (d/2,)

    # construct angles to rotate by
    m = jnp.expand_dims(m, axis=1)           # (seq_len, 1)
    angles = m * jnp.repeat(theta, 2)        # (seq_len, d)

    rot1 = jnp.cos(angles)                   # (d,)
    rot2 = jnp.sin(angles)                   # (d,)

    # swap odd indices down by 1 and negate (in embedding dim)
    # this is probably slow, but we only have to do it once
    weaved = x.at[:, ::2].set(-x[:, 1::2])
    weaved = weaved.at[:, 1::2].set(x[:, ::2])

    return (x*rot1 + weaved*rot2).astype(jnp.bfloat16)

# multi-head and batched multi-head
# MH: (T, d), (T,) -> (T, d) |==> (H, T, d), (T,) -> (H, T, d)
# BH: MH |==> (B, H, T, d), (B, T) -> (B, H, T, d)
apply_rope_mh = jax.vmap(apply_RoPE, in_axes=(0, None, None), out_axes=0)
apply_rope_batch = jax.vmap(apply_rope_mh, in_axes=(0, 0, None), out_axes=0)


if __name__ == "__main__":


    # seq_len = 10, d = 4
    d = 4
    T = 9
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (T, d))
    print(x)

    desc = init_RoPE(d)

    print("=== Result ===")
    
    print(apply_RoPE(x, jnp.arange(T), desc))