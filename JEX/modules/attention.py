import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl       # triton-like API

from typing import NamedTuple

# replacing upper triangular with this -> zero out in softmax
CAUSAL_CONST = -jnp.inf


class AtnDescriptor(NamedTuple):
    """
    Contains params for attention layer (MHA, MQA)

    d_model: model dimension
    H_q: number of heads for Q
    H_k: number of heads for K
    H_v: number of heads for V

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


# Fallbacks, probably use flash attention in practice
def MHA(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, atn_descriptor: AtnDescriptor):
    """
    Vanilla multi-head attention
    """
    d_k = atn_descriptor.q_proj.shape[-1]
    
    q = jnp.einsum('Bnd,Hdk->BHnk', q, atn_descriptor.q_proj)           # contraction along d_k
    k = jnp.einsum('Bnd,Hdk->BHnk', k, atn_descriptor.k_proj)           # contraction along d_k
    v = jnp.einsum('Bnd,Hdv->BHnv', v, atn_descriptor.v_proj)           # contraction along d_v

    # scaled dot product attention (naive)
    # TODO: flash attention
    qk = jnp.einsum('BHnk,BHmk->BHnm', q, k) / jnp.sqrt(d_k)
    attn_weights = jax.nn.softmax(qk, axis=-1)

    output = jnp.einsum('BHnm,BHnv->BHmv', attn_weights, v)
    
    return output


def MQA(q: jnp.array, atn_descriptor: AtnDescriptor, mid_fn=None):
    """
    Multi-query (self) attention (1 head for K and V, multiple heads for Q)

    Note: number of dimensions in a head is implicit in the shape of k and v_proj
    - We'll denote this as d_k and d_v

    Args:
        q: (B, T, d_model)
        atn_descriptor: attention weights
        mid_fn: intermediate function to transform Q/K/V before attention
    """
    d_k = atn_descriptor.q_proj.shape[-1]
    T = q.shape[1]

    # project: (B, H, T, d_model) -> (B, H, T, d_k)
    # contraction along d_model (m)
    q_p = jnp.einsum('BTm,Hkm->BHTk', q, atn_descriptor.q_proj)
    k_p = jnp.einsum('BTm,Hkm->BHTk', q, atn_descriptor.k_proj)
    v_p = jnp.einsum('BTm,Hvm->BHTv', q, atn_descriptor.v_proj)

    # accomodate some intermediate function (usually scaling/positional encoding)
    q_p, k_p, v_p = mid_fn(q_p, k_p, v_p) if mid_fn is not None else (q_p, k_p, v_p)

    # broadcast K to V, recall K has 1 head
    # Q: (B, H, T, d_model), K: (B, 1, T, d_model) -> (B, H, T, T)
    QK = jnp.tril(jnp.einsum('BHTk,BHRk->BHTR', q_p, k_p))

    # mask out upper triangular (causal)
    # (1, 1, T, T) 
    mask = (jnp.triu(jnp.full((T, T), CAUSAL_CONST), 1))[None, None, ...]
    QK = QK + mask
    attn_weights = jax.nn.softmax(QK, axis=-1)

    # apply attention weights to V
    # (B, H, T, T) x (B, H, T, d_v) -> (B, H, T, d_v)
    o = attn_weights @ v_p
 
    # concatenate heads (B, H, T, d_v) -> (B, T, H*d_v)
    o = o.transpose((0, 2, 1, 3)).reshape((q.shape[0], q.shape[1], -1))

    # project output (B, T, H*d_v) @ (d_model, H*d_v) -> (B, T, d_model)
    o_p = jnp.einsum('BTv,mv->BTm', o, atn_descriptor.o_proj)
    return o_p 


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
    res = MQA(x, desc)
    print(res.shape)

    # testing jit compatability
    MQA_jit = jax.jit(MQA)
    res_jit = MQA_jit(x, desc)

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
    res = MQA(x, desc)[:, :, 0].flatten()

    print("=== Correctness ===")
    print("Match:", jnp.allclose(res, jnp.array([2, 4, 6, 8])))
    print(f"frobenius: {jnp.linalg.norm((res - jnp.array([2,4,6,8])))}")

    # test Multi query case
    q_proj = jnp.ones((2, 1, d_model))
    desc = AtnDescriptor(q_proj, k_proj, v_proj, o_proj)
    res = MQA(x, desc)