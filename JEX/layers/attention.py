import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl       # triton-like API

from typing import NamedTuple

# replacing upper triangular with this -> zero out in softmax
MASK_CONST = -jnp.inf


class AtnDescriptor(NamedTuple):
    """
    Contains params for attention layer (MHA, MQA)

    Shape conventions:
        d_model: model dimension
        d_q: query dimension
        d_k: key dimension
        d_v: value dimension
        H_q: number of heads for Q
        H_k: number of heads for K
        H_v: number of heads for V

    Attributes:
        q_proj: project residual to query space
        k_proj: project residual to key space
        v_proj: project residual to value space
        o_proj: project output to residual stream

    See: https://transformer-circuits.pub/2021/framework/index.html
    """

    q_proj: jnp.ndarray     # (H_q, d_k, d_model)
    k_proj: jnp.ndarray     # (H_k, d_k, d_model)
    v_proj: jnp.ndarray     # (H_v, d_v, d_model)
    o_proj: jnp.ndarray     # (d_model, H_v*d_v)


def apply_qkv_proj(x: jnp.array, atn_descriptor: AtnDescriptor):
    """
    Apply qkv projection to input. Useful for when we
    want to retrieve intermediate Q, K, V directly for caching. 
    
    Assume self attention.

    Args:
        x: (B, T, d_model)
        atn_descriptor: attention weights

    Returns:
        q: (B, H_q, T, d_q)
        k: (B, H_k, T, d_k)
        v: (B, H_v, T, d_v)
    """

    # contraction along d_model (m)
    q_p = jnp.einsum('BTm,Hkm->BHTk', x, atn_descriptor.q_proj)
    k_p = jnp.einsum('BTm,Hkm->BHTk', x, atn_descriptor.k_proj)
    v_p = jnp.einsum('BTm,Hvm->BHTv', x, atn_descriptor.v_proj)

    return q_p, k_p, v_p


# Fallbacks, probably use flash attention in practice
def MQA(q_p: jnp.array, 
        k_p: jnp.array,
        v_p: jnp.array, 
        atn_descriptor: AtnDescriptor):
    """
    Multi-query (self) attention (1 head for K and V, multiple heads for Q)
    Also covers multi-head case

    Note: head dimensionality is implicit in the shape of k and v_proj
    - We denote this as d_k and d_v

    Args:
        q: (B, H_q, T, d_q)
        k: (B, H_k, T, d_k)
        v: (B, H_v, T, d_v)
        atn_descriptor: attention weights
        mid_fn: intermediate function to transform Q/K/V before attention
    """
    B = q_p.shape[0]
    T = q_p.shape[2]

    # broadcast K to V (recall K has 1 head in MQA)
    # Q: (B, H_q, T, d_model), K: (B, H_k, T, d_model) -> (B, H, T, T)
    QK = jnp.tril(jnp.einsum('BHTk,BJRk->BHTR', q_p, k_p))

    # mask out upper triangular (causal) (1, 1, T, T) 
    mask = (jnp.triu(jnp.full((T, T), MASK_CONST), 1))[None, None, ...]
    QK = QK + mask
    attn_weights = jax.nn.softmax(QK, axis=-1)

    # apply attention weights to V
    # (B, H, T, T) x (B, H, T, d_v) -> (B, H, T, d_v)
    o = attn_weights @ v_p
 
    # concatenate heads (B, H, T, d_v) -> (B, T, H*d_v)
    o = o.transpose((0, 2, 1, 3)).reshape((B, T, -1))

    # project output (B, T, H*d_v) @ (d_model, H*d_v) -> (B, T, d_model)
    return jnp.einsum('BTv,mv->BTm', o, atn_descriptor.o_proj)


def MQA_cached(q: jnp.array, 
               K_cache: jnp.array, 
               V_cache: jnp.array,
               T: int,
               atn_descriptor: AtnDescriptor, 
               mid_fn=None
               ):
    """
    MQA+MHA but with a KV cache. Meant for streaming
        - H=1 in the MQA case

    Assume K and V cache are prefilled properly
    Args:
        q: (B, 1, d_model)
        K_cache: (B, H, T, d_k)
        V_cache: (B, H, T, d_v), V before attention reweighting
        T: index of q
        atn_descriptor: attention weights
        mid_fn: intermediate function to transform Q/K/V before attention

    """
    # TODO: prevent unecessary computations? This is a dumb implementation
    
    # project: (B, H, 1, d_model) -> (B, H, d_k)
    # contraction along d_model (m)
    q_p = jnp.einsum('BTm,Hkm->BHk', q, atn_descriptor.q_proj)
    k_p = jnp.einsum('BTm,Hkm->BHk', q, atn_descriptor.k_proj)
    v_p = jnp.einsum('BTm,Hvm->BHv', q, atn_descriptor.v_proj)

    q_p, k_p, v_p = mid_fn(q_p, k_p, v_p, T) if mid_fn is not None else (q_p, k_p, v_p)

    k_p = K_cache.at[:, :, T].set(k_p)
    v_p = V_cache.at[:, :, T].set(v_p)

    # compute lowest row of attention matrix
    QK = jnp.einsum('BHk,BHTk->BHT', q_p, k_p)
    attn_weights = jax.nn.softmax(QK, axis=-1).at[:, :, None, :].get()

    # apply attention weights to V
    # (B, H, 1, T) x (B, H, T, d_v) -> (B, H, T, d_v)
    o = attn_weights @ v_p
 
    # concatenate heads (B, H, T, d_v) -> (B, T, H*d_v)
    o = o.transpose((0, 2, 1, 3)).reshape((q.shape[0], q.shape[1], -1))

    # project output (B, T, H*d_v) @ (d_model, H*d_v) -> (B, T, d_model)
    o_p = jnp.einsum('BTv,mv->BTm', o, atn_descriptor.o_proj)

    return o_p, k_p, v_p



def GQA(q: jnp.array, atn_descriptor):
    """
    Grouped query attention
    """

    # TODO: implement
    pass


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    # d_query, d_key, d_value = 2
    # d_model = 8
    # T = 10

    B = 5
    T = 10
    d_model = 8
    d_k = 2
    d_value = d_k
    d_v = 2
    H = 3
    x = jnp.arange(B*T*d_model).reshape((B, T, d_model))
    print("input:", x.shape)

    # seq_len = 10, d_model = 4
    q_proj = jax.random.normal(rng, (H, d_k, d_model))
    k_proj = jax.random.normal(rng, (1, d_k, d_model))
    v_proj = jax.random.normal(rng, (1, d_v, d_model))
    o_proj = jax.random.normal(rng, (d_model, d_v*H))

    desc = AtnDescriptor(q_proj, k_proj, v_proj, o_proj)

    print("=== Result ===")
    q, k, v = apply_qkv_proj(x, desc)
    res = MQA(q, k, v, desc)
    print(res.shape)

    # testing jit compatability
    MQA_jit = jax.jit(MQA)
    res_jit = MQA_jit(q, k, v, desc)

    print(f"JIT parity: {jnp.allclose(res, res_jit)}")
    print(f"frobenius: {jnp.linalg.norm((res - res_jit).flatten())}")

    # test correctness
    T = 4
    d_model = 2

    # B = 1, T = 4, d_model = 2
    x = jnp.arange(1, 5)[None, :, None]
    x = jnp.repeat(x, d_model, axis=-1)

    q_proj = jnp.ones((1, 1, d_model))
    k_proj = jnp.ones((1, 1, d_model))
    v_proj = jnp.ones((1, 1, d_model))
    o_proj = jnp.ones((d_model, 2))

    desc = AtnDescriptor(q_proj, k_proj, v_proj, o_proj)
    q, k, v = apply_qkv_proj(x, desc)

    res = MQA(q, k, v, desc)[:, :, 0].flatten()

    print("=== Correctness ===")
    print("Match:", jnp.allclose(res, jnp.array([2, 4, 6, 8])))
    print(f"frobenius: {jnp.linalg.norm((res - jnp.array([2,4,6,8])))}")

    # test Multi query case
    q_proj = jnp.ones((2, 1, d_model))
    desc = AtnDescriptor(q_proj, k_proj, v_proj, o_proj)
    res = MQA(q, k, v, desc)

    # Cached case
    print("=== Cached ===")
    K_cache = jnp.array([[
        [
            [2],
            [4],
            [6],
            [0]
        ]
    ]])
    V_cache = jnp.array([[
        [
            [2],
            [4],
            [6],
            [0]
        ]
    ]])

    q = jnp.array([[[4, 4]]])

    res, K_cache, V_cache = MQA_cached(q, K_cache, V_cache, 3, desc)

    c = 3