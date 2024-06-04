# 2.5 billion parameter model for now
# bf16 weights, so min 5GB in VRAM
# TODO: profile memory usage

# https://github.com/google-deepmind/gemma/tree/main
# ^google's implementation in flax

import jax
import jax.numpy as jnp
from modules import attention, activations, norm, pos_embedding
from typing import NamedTuple, Optional
from functools import partial


class GemmaDescriptor(NamedTuple):
    """
    Contains params for GEMMA block
    """
    d_model: int                          # activation space dimension
    n_heads: int                          # num heads in MHA or number of query heads in MQA
    dropout: float                        # p(drop), probably not needed for inference
    theta: jnp.array                      # minimum rotary angles (d_model / 2,)
    embed: jnp.array                      # embedding matrix (vocab_size, d_model)
    unembed: Optional[jnp.array] = None   # None = shared with embedding
    blocks: list[NamedTuple] = None       # decoder blocks
    kv_cache: Optional[jnp.array] = None  # TODO: key-value cache for MQA


class GemmaBlockDescriptor(NamedTuple):
    """
    Contains params for GEMMA block
    """
    up_proj: jnp.array
    gate_proj: jnp.array
    down_proj: jnp.array
    scale: jnp.array
    atn: attention.AtnDescriptor
    prenorm: norm.RMSNormDescriptor
    postnorm: norm.RMSNormDescriptor

GeGLU = partial(activations.GLU, b1=0.0, b2=0.0)

def embed_tokens(tokens: jnp.array, desc: GemmaDescriptor):
    """
    Embeds tokens into activation space and apply positional encoding

    Args:
        tokens: (B, N) sequence of token ids
        desc: GemmaDescriptor
    """

    x = jnp.take(desc.embed, tokens, axis=0)
    x = x + pos_embedding.apply_RoPE(desc.theta, x)
    return x


def unembed(x: jnp.array, desc: GemmaDescriptor):
    """
    convert from activation space to logits

    Args:
        x: (B, N, d_model)
        desc: GemmaDescriptor
    """

    if desc.unembed is None:
        return x @ desc.embed.T
    return x @ desc.unembed.T


def apply_gemma_block(x: jnp.array, desc: GemmaBlockDescriptor):
    """
    GEMMA block
    """
    # pre-attn norm
    x = norm.apply_RMSNorm(desc.prenorm, x)

    # self attention
    v = attention.MQA(x, desc.atn)

    # add to residual stream, apply norm
    x = x + v
    x = norm.apply_RMSNorm(desc.postnorm, x)

    # apply MLP (B, T, d_model) -> (B, T, d_g)
    x = GeGLU(x, desc.gate_proj, desc.up_proj)
    
    # project back to d_model
    return x @ desc.down_proj.T


def gemma_forward(x: jnp.array, desc: GemmaDescriptor):
    """
    One forward pass of GEMMA 

    Args:
        x: (B, N) sequence of token ids
        desc: GemmaDescriptor
    """

    x = embed_tokens(x, desc)

    # JAX unrolls this loop when jitting
    for block in desc.blocks:
        x = apply_gemma_block(x, block)

    return unembed(x, desc)


if __name__ == "__main__":
    # test
    pass