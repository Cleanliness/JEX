import jax
import jax.numpy as jnp

from typing import NamedTuple


class AtnDescriptor(NamedTuple):
    """
    Contains params for MHA layer 
    """
    q_proj: jnp.ndarray     # (H, d_model, d_k)
    k_proj: jnp.ndarray     # (H, d_model, d_k)
    v_proj: jnp.ndarray     # (H, d_model, d_v)


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


def MQA(q: jnp.array, k: jnp.array, v: jnp.array, atn_descriptor: AtnDescriptor):
    """
    Multi-query attention (1 head for K and V, multiple heads for Q)
    """
    d_k = atn_descriptor.q_proj.shape[-1]
    
    q = jnp.einsum('Bnd,Hdk->BHnk', q, atn_descriptor.q_proj)           # contraction along d_k
    k = jnp.einsum('Bnd,Hdk->BHnk', k, atn_descriptor.k_proj)           # contraction along d_k
    v = jnp.einsum('Bnd,Hdv->BHnv', v, atn_descriptor.v_proj)           # contraction along d_v
