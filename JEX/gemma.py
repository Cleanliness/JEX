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

from time import time


class GemmaDescriptor(NamedTuple):
    """
    Contains state of a GEMMA model
    """
    d_model: int                          # activation space dimension
    n_heads: int                          # num heads in MHA or number of query heads in MQA
    dropout: float                        # p(drop), probably not needed for inference
    rope: pos_embedding.RoPEDescriptor    # RoPE angles
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
    atn: attention.AtnDescriptor
    prenorm: norm.RMSNormDescriptor
    postnorm: norm.RMSNormDescriptor

# ==== gemma specific functions ====
def scale_q(q, k, v, rd):
    """
    *multiply* query by sqrt(d_k) before dot product.
    NOT divide, multiply.

    deepmind does this for some godforsaken reason
    """
    d_k = k.shape[-1]
    B = q.shape[0]
    T = q.shape[2]
    m = jnp.repeat(jnp.arange(T), B).reshape(B, T)
    q = pos_embedding.apply_rope_batch(q, m, rd)
    k = pos_embedding.apply_rope_batch(k, m, rd)

    return q * jnp.sqrt(d_k), k, v


gemma_GeGLU = partial(activations.GLU, b1=0.0, b2=0.0)

# ==== gemma forward pass ====
def embed_tokens(tokens: jnp.array, desc: GemmaDescriptor):
    """
    Embeds tokens into activation space and apply positional encoding

    Args:
        tokens: (B, N) sequence of token ids
        desc: GemmaDescriptor
    """

    x = jnp.take(desc.embed, tokens, axis=0)
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


def apply_gemma_block(x: jnp.array, desc: GemmaBlockDescriptor, mid_fn=None):
    """
    GEMMA block
    """
    # pre-attn norm
    x = norm.apply_RMSNorm(desc.prenorm, x)

    # self attention
    v = attention.MQA(x, desc.atn, mid_fn)

    # add to residual stream, apply norm
    x = x + v
    x = norm.apply_RMSNorm(desc.postnorm, x)

    # apply MLP (B, T, d_model) -> (B, T, d_g)
    x = gemma_GeGLU(x, desc.gate_proj, desc.up_proj)
    
    # project back to d_model
    return x @ desc.down_proj.T


def construct_gemma_forward(desc: GemmaDescriptor):
    """
    create jittable forward pass of GEMMA 

    Args:
        x: (B, N) sequence of token ids
        desc: GemmaDescriptor
    """

    gemma_scale_q = partial(scale_q, rd=desc.rope)
    block_fn = partial(apply_gemma_block, mid_fn=gemma_scale_q)
    def fwd(x: jnp.array):
        x = embed_tokens(x, desc)

        # JAX unrolls this loop when jitting
        for block in desc.blocks:
            x = apply_gemma_block(x, block, mid_fn=gemma_scale_q)

        # do with scan instead
        # x = jax.lax.scan(block_fn, x, desc.blocks)
        return unembed(x, desc)

    return fwd


def init_gemma(d_model: int,
               d_up: int,
               H_dim: int,
               n_heads: int, 
               vocab_size: int, 
               n_blocks: int, 
               theta_max: int=10000,
               atn_type: str="MHA"):
    """
    Initialize a GEMMA model

    Args:
        d_model: activation space dimension
        d_up: dimension of gate MLP
        H_dim: number of dimensions in an attention head
        n_heads: num heads in MHA or number of query heads in MQA
        vocab_size: size of vocabulary
        n_blocks: number of GEMMA blocks
        theta_max: max rotation period for subspaces
        atn_type: attention type (MHA or MQA)
    """

    assert d_model % 2 == 0, "embedding dimension must be even"

    embed = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, d_model)).astype(jnp.bfloat16)
    rope = pos_embedding.init_RoPE(H_dim, theta_max)
    rng = jax.random.PRNGKey(0)

    blocks = []
    for _ in range(n_blocks):
        # MLP projections
        up_proj = jax.random.normal(rng, (d_up, d_model)).astype(jnp.bfloat16)
        gate_proj = jax.random.normal(rng, (d_up, d_model)).astype(jnp.bfloat16)
        down_proj = jax.random.normal(rng, (d_model, d_up)).astype(jnp.bfloat16)

        # pre and post norm
        scale1 = jax.random.normal(rng, (d_model,)).astype(jnp.bfloat16)
        scale2 = jax.random.normal(rng, (d_model,)).astype(jnp.bfloat16)

        # attention and normalization weights
        if atn_type == "MQA":
            k_heads = 1
            v_heads = 1
        else:
            k_heads = n_heads
            v_heads = n_heads

        q_proj = jax.random.normal(rng, (n_heads, H_dim, d_model)).astype(jnp.bfloat16)
        k_proj = jax.random.normal(rng, (k_heads, H_dim, d_model)).astype(jnp.bfloat16)
        v_proj = jax.random.normal(rng, (v_heads, H_dim, d_model)).astype(jnp.bfloat16)
        o_proj = jax.random.normal(rng, (d_model, n_heads*H_dim)).astype(jnp.bfloat16)

        atn = attention.AtnDescriptor(q_proj, k_proj, v_proj, o_proj)
        norm1 = norm.RMSNormDescriptor(scale1)
        norm2 = norm.RMSNormDescriptor(scale2)

        blocks.append(
            GemmaBlockDescriptor(
                up_proj=up_proj,
                gate_proj=gate_proj,
                down_proj=down_proj,
                atn=atn,
                prenorm=norm1,
                postnorm=norm2
            )
        )


    return GemmaDescriptor(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        rope=rope,
        embed=embed,
        blocks=blocks
    )


def count_params(desc: GemmaDescriptor):
    """
    Count number of parameters in GEMMA model
    """
    n_params = desc.embed.size
    for block in desc.blocks:
        n_params += block.up_proj.size
        n_params += block.gate_proj.size
        n_params += block.down_proj.size
        n_params += block.atn.q_proj.size
        n_params += block.atn.k_proj.size
        n_params += block.atn.v_proj.size
        n_params += block.atn.o_proj.size
        n_params += block.prenorm.scale.size
        n_params += block.postnorm.scale.size

    return n_params


if __name__ == "__main__":
    # model params 
    d_model = 2048
    d_up = 4096 
    H_dim = 512
    n_heads = 4
    vocab_size = 100000
    n_blocks = 8

    # test params
    runs = 1000

    gem = init_gemma(d_model, d_up, H_dim, n_heads, vocab_size, n_blocks)

    print(f"Created model with {count_params(gem)} params")
    # dummy batch of 1 sequence of 3 tokens
    x = jnp.array([[1, 2, 3],])
    fwd = construct_gemma_forward(gem)

    print("Compiling forward pass")
    start = time()
    gemma_forward = jax.jit(fwd)
    c = gemma_forward(x).block_until_ready()    # warmup, JAX is lazy
    print("Compiled in ", time() - start, "seconds")

    print(f"timing forward pass on {runs} runs")
    t_delta_sum = 0

    for i in range(runs):
        start = time()
        c = gemma_forward(x).block_until_ready()
        t_delta_sum += time() - start
        x += 1
    
    delta_avg = t_delta_sum / runs
    print("Average time: ", delta_avg, "seconds")
    
    # this probably sucks because no KV cache
    print("Tokens/second: ", 1 / delta_avg)

