# 2.5 billion parameter model for now
# bf16 weights, so min 5GB in VRAM
# TODO: profile memory usage

# https://github.com/google-deepmind/gemma/tree/main
# ^google's implementation in flax

import jax
import jax.ad_checkpoint
import jax.numpy as jnp
from modules import attention, activations, norm, pos_embedding
from typing import NamedTuple, Optional
from functools import partial

from time import time
from read_model import load_model 
from tokenizer import load_tokenizer, encode_str, decode_tk
from modules.pos_embedding import init_RoPE


class GemmaDescriptor(NamedTuple):
    """
    Contains state of a GEMMA model
    """
    d_model: int                          # activation space dimension
    n_heads: int                          # num heads in MHA or number of query heads in MQA
    rope: pos_embedding.RoPEDescriptor    # RoPE angles
    embed: jnp.array                      # embedding matrix (vocab_size, d_model)
    final_norm: norm.RMSNormDescriptor    # Final layer normalization
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
gemma_GeGLU = partial(activations.GLU, b1=0.0, b2=0.0)
def qk_rope(q, k, v, rd):
    """
    Apply rope to queries and keys
    """
    B = q.shape[0]
    T = q.shape[2]
    m = jnp.tile(jnp.arange(T), B).reshape(B, T)
    return pos_embedding.apply_rope_batch(q, m, rd), pos_embedding.apply_rope_batch(k, m, rd), v

# ==== gemma forward pass ====
def embed_tokens(tokens: jnp.array, emb: jnp.array):
    """
    Embeds tokens into activation space and apply positional encoding

    Args:
        tokens: (B, N) sequence of token ids
        desc: GemmaDescriptor
    """

    x = jnp.take(emb, tokens, axis=0)
    return x


def apply_gemma_block(x: jnp.array, desc: GemmaBlockDescriptor, mid_fn=None):
    """
    GEMMA block
    """
    # pre-attn norm
    x = norm.apply_RMSNorm(desc.prenorm, x)

    # self attention
    v = attention.MQA(x, desc.atn, mid_fn=mid_fn)

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

    qk_rope_fn = partial(qk_rope, rd=desc.rope)
    def fwd(x: jnp.array, gdesc: GemmaDescriptor):
        # deepmind multiplies by sqrt(d_model) for some reason
        x = embed_tokens(x, gdesc.embed)*jnp.sqrt(gdesc.d_model).astype(jnp.bfloat16)

        # JAX unrolls this loop when jitting
        # lax.scan is slower
        for block in gdesc.blocks:
            x = apply_gemma_block(x, block, mid_fn=qk_rope_fn)
        
        x = norm.apply_RMSNorm(gdesc.final_norm, x)

        # take last token and unembed
        return x.at[:, -1].get() @ gdesc.embed.T

    return fwd


def init_gemma(d_model: int,
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
        blocks=blocks
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


if __name__ == "__main__":
    # model params 
    d_model = 2048
    n_blocks = 18
    d_up = 16384
    H_dim = 256
    n_heads = 8
    vocab_size = 256128     # don't go past this

    TEST_STR = r"Q: What's the capital of France?\nA:"

    # test params
    runs = 100

    tk = load_tokenizer(r"/media/roy/TOSHIBA EXT/models/gemma-2b/tokenizer.model")

    # this is identical in structure
    # gem = init_gemma(d_model, d_up, H_dim, n_heads, vocab_size, n_blocks)

    # loading the real thing
    gem = load_gemma_model()

    seq = encode_str(TEST_STR, tk)
    print(f"parameter count:{count_params(gem)}")
    fwd = jax.checkpoint(construct_gemma_forward(gem))

    # shitty correctness test
    for i in range(3):
        x = jnp.expand_dims(jnp.array(seq), 0)
        print("Compiling forward pass")
        start = time()
        gemma_forward = jax.jit(fwd)
        c = gemma_forward(x, gem).block_until_ready()    # warmup, JAX is lazy
        print("Compiled in ", time() - start, "seconds")

        res = jnp.argmax(c, axis=-1).tolist()
        seq += res
        print(f"response: {decode_tk(res, tk)}")

    print(f"timing forward pass on {runs} runs")
    t_delta_sum = 0

    for i in range(runs):
        start = time()
        c = gemma_forward(x, gem).block_until_ready()
        t_delta_sum += time() - start
        x += 1
    
    delta_avg = t_delta_sum / runs
    print("Average time: ", delta_avg, "seconds")
    
    # this probably sucks because no KV cache
    print("Tokens/second: ", 1 / delta_avg)

