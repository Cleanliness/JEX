# 2.5 billion parameter model for now
# bf16 weights, so min 5GB in VRAM
# TODO: profile memory usage

# https://github.com/google-deepmind/gemma/tree/main
# ^google's implementation in flax


import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional
from functools import partial

from time import time

# intra-package references: https://docs.python.org/3/tutorial/modules.html
# NOTE: relative imports only work inside packages
from .read_model import load_model 
from .layers.pos_embedding import init_RoPE
from .layers import attention, activations, norm, pos_embedding


class GemmaDescriptor(NamedTuple):
    """
    Contains state of a Gemma model
    """
    d_model: int                          # residual stream dimension
    n_heads: int                          # num heads in MHA or number of query heads in MQA
    rope: pos_embedding.RoPEDescriptor    # RoPE angles
    embed: jnp.array                      # embedding matrix (vocab_size, d_model)
    final_norm: norm.RMSNormDescriptor    # Final layer normalization
    unembed: Optional[jnp.array] = None   # None = shared with embedding
    blocks: list[NamedTuple] = None       # decoder blocks


class GemmaBlockDescriptor(NamedTuple):
    """
    Contains params for Gemma block
    """
    up_proj: jnp.array
    gate_proj: jnp.array
    down_proj: jnp.array
    atn: attention.AtnDescriptor
    prenorm: norm.RMSNormDescriptor
    postnorm: norm.RMSNormDescriptor


# ==== gemma specific functions ====
gemma_GeGLU = partial(activations.GLU, b1=0.0, b2=0.0)
approx_gelu = partial(jax.nn.gelu, approximate=True)


def qk_rope(q, k, v, rd, t: int=None):
    """
    Apply rope to queries and keys after qkv projection
    """
    B = q.shape[0]
    T = q.shape[2]
    d_model = q.shape[-1]

    if t is None:
        m = jnp.tile(jnp.arange(T), B).reshape(B, T)
    else:
        m = jnp.tile(jnp.array([t]), B).reshape(B, 1)       

    q_rope = pos_embedding.apply_rope_batch(q, m, rd)
    k_rope = pos_embedding.apply_rope_batch(k, m, rd)
    return q_rope / jnp.sqrt(d_model), k_rope, v


# ==== gemma forward pass ====
def embed_tokens(tokens: jnp.array, emb: jnp.array):
    """
    Embeds tokens into activation space and apply positional encoding

    Args:
        tokens: (B, T) sequence of token ids
        desc: GemmaDescriptor
    """

    x = (jnp.take(emb, tokens, axis=0) * jnp.sqrt(emb.shape[-1]).astype(jnp.bfloat16))
    return x


def apply_gemma_block(x: jnp.array, 
                      desc: GemmaBlockDescriptor,
                      KV_size: int=256,
                      mid_fn=None,
                      K_cache: jnp.array=None,
                      V_cache: jnp.array=None,
                      t: int=None):
    """
    GEMMA block forward pass
    """
    B = x.shape[0]
    H = desc.atn.k_proj.shape[0]
    d_k = desc.atn.k_proj.shape[1]
    d_v = desc.atn.v_proj.shape[1]

    # pre-attn norm
    x_norm = norm.apply_RMSNorm(desc.prenorm, x)
    q, k, v = attention.apply_qkv_proj(x_norm, desc.atn)
    q, k, v = mid_fn(q, k, v, t=t) if mid_fn is not None else (q, k, v)

    # self attention
    # no cache
    if t is None:
        o = attention.MQA(q, k, v, desc.atn)
        T_v = v.shape[2]
        K_cache = jnp.zeros((B, H, KV_size, d_k)).at[:, :, :T_v].set(k)
        V_cache = jnp.zeros((B, H, KV_size, d_v)).at[:, :, :T_v].set(v)


    # cache exists, only compute a single timestep
    else:
        o, K_cache, V_cache = attention.MQA_cached(x_norm, K_cache, V_cache, t, desc.atn)

    # add to residual stream, apply norm
    x += o
    post_atn = norm.apply_RMSNorm(desc.postnorm, x)

    # apply MLP (B, T, d_model) -> (B, T, d_g)
    post_atn = gemma_GeGLU(post_atn, desc.gate_proj, desc.up_proj, gate_fn=approx_gelu)
    post_atn = post_atn @ desc.down_proj.T 

    # add back to residual stream
    return x + post_atn, K_cache, V_cache


def construct_gemma_forward(desc: GemmaDescriptor):
    """
    create jittable forward pass of GEMMA. Computes whole atn matrix in one go

    Args:
        desc: GemmaDescriptor
    """

    qk_rope_fn = partial(qk_rope, rd=desc.rope)

    # TODO: replace hardcoded KV size
    blk = partial(apply_gemma_block, KV_size=256, mid_fn=qk_rope_fn)

    def fwd(x: jnp.array, gdesc: GemmaDescriptor, kv: tuple[list[jnp.array]]=None,  t: int=None):
        # deepmind multiplies by sqrt(d_model) for some reason
        x = embed_tokens(x, gdesc.embed).astype(jnp.bfloat16)

        if kv is None:
            k_cache = [None for _ in range(len(gdesc.blocks))]
            v_cache = [None for _ in range(len(gdesc.blocks))]
        else:
            k_cache, v_cache = kv

        # JAX unrolls this loop when jitting
        # lax.scan is slower
        for i, block in enumerate(gdesc.blocks):
            x, k, v = blk(x, block, K_cache=k_cache[i], V_cache=v_cache[i], t=t)

            # this doesn't actually mutate the pytree, seems to copy it
            # we need to return the descriptor at the end for statefulness
            k_cache[i] = k
            v_cache[i] = v

        x = norm.apply_RMSNorm(gdesc.final_norm, x)

        # pytrees are copied on return for jit!!!
        # only return what gets modified to save memory
        kv = (k_cache, v_cache)

        # unembed
        return x.at[:, -1].get() @ gdesc.embed.T, kv

    return fwd


def make_gemma(d_model: int,
               d_up: int,
               H_dim: int,
               n_heads: int, 
               vocab_size: int, 
               n_blocks: int, 
               theta_max: int=10000,
               atn_type: str="MQA"):
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

    final_norm = norm.RMSNormDescriptor(jax.random.normal(rng, (d_model,)).astype(jnp.bfloat16))

    return GemmaDescriptor(
        d_model=d_model,
        n_heads=n_heads,
        rope=rope,
        embed=embed,
        final_norm=final_norm,
        blocks=blocks,
    )


def load_gemma_model():
    """
    Load Gemma 2B
    """
    params = load_model()

    embed = params["model.embed_tokens.weight"]
    final_norm = norm.RMSNormDescriptor(params["model.norm.weight"])

    blocks = []
    print("loading blocks")

    for i in range(18):
        atn = attention.AtnDescriptor(
            q_proj=jnp.reshape(params[f"model.layers.{i}.self_attn.q_proj.weight"], (8, 256, 2048)),
            k_proj=jnp.reshape(params[f"model.layers.{i}.self_attn.k_proj.weight"], (1, 256, 2048)),
            v_proj=jnp.reshape(params[f"model.layers.{i}.self_attn.v_proj.weight"], (1, 256, 2048)),
            o_proj=params[f"model.layers.{i}.self_attn.o_proj.weight"],
        )
        prenorm = norm.RMSNormDescriptor(params[f"model.layers.{i}.input_layernorm.weight"])
        postnorm = norm.RMSNormDescriptor(params[f"model.layers.{i}.post_attention_layernorm.weight"])
        curr = GemmaBlockDescriptor(
            up_proj=params[f"model.layers.{i}.mlp.up_proj.weight"],
            gate_proj=params[f"model.layers.{i}.mlp.gate_proj.weight"],
            down_proj=params[f"model.layers.{i}.mlp.down_proj.weight"],
            atn=atn,
            prenorm=prenorm,
            postnorm=postnorm
        )

        blocks.append(curr)
    
 
    return GemmaDescriptor(
        d_model=2048,
        n_heads=8,
        rope=init_RoPE(256),    # head dim is 256
        embed=embed,
        final_norm=final_norm,
        blocks=blocks,
    )


def count_params(desc: GemmaDescriptor):
    """
    Count number of parameters in GEMMA model
    """
    embed_params = desc.embed.size
    n_params = desc.final_norm.scale.size
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

    return n_params, embed_params
