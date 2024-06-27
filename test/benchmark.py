# Hacky way to access
# TODO: turn this into an installable pkg
from pathlib import Path
import sys
from time import time
 
# directory reach
directory = Path(__file__)
 
# setting path
sys.path.append(str(directory.parent.parent))
print(sys.path)

from JEX import gemma, read_model, tokenizer

import jax
import jax.numpy as jnp

# Promp processing
# Metrics: tok/s, MFU
def benchmark_pps():
    pass


# decoding speed @ KV cache size
# Metrics: tok/s, MFU
def benchmark_decode():
    pass


# test memory usage over time
def test_mem():
    pass


if __name__ == "__main__":
    # model params 
    d_model = 2048
    n_blocks = 18
    d_up = 16384
    H_dim = 256
    n_heads = 8
    vocab_size = 256128     # don't go past this

    import matplotlib.pyplot as plt
    import numpy as np

    correct_logits = jnp.array(np.load("/home/roy/Documents/Programming/JEMMA/test/gemma-2b.npz")["logits"])
    correct_hiddens = jnp.array(np.load("/home/roy/Documents/Programming/JEMMA/test/gemma-2b-layers.npz")["outs"])
    TEST_STR = r"Q: What's the capital of France?\nA:"

    # test params
    runs = 100

    gem = gemma.make_gemma(d_model, d_up, H_dim, n_heads, vocab_size, n_blocks)
    test_x = jnp.ones((1, 1), dtype=jnp.int32)
    fwd = jax.jit(gemma.construct_gemma_forward(gem))

    k = [jnp.ones((1, 1, 256, 256)) for _ in range(18)]
    v = [jnp.ones((1, 1, 256, 256)) for _ in range(18)]

    kv = (k, v)

    # compile warmup
    res, kv = fwd(test_x, gem, kv, t=128)
    res.block_until_ready()

    # ==== Benchmarking ====
    print(f"timing forward pass on {runs} runs")
    t_delta_sum = 0

    for i in range(runs):
        start = time()
        res, kv = fwd(test_x, gem, kv, t=128)
        res.block_until_ready()
        t_delta_sum += time() - start
        test_x += 1
    
    delta_avg = t_delta_sum / runs
    print("Average time: ", delta_avg, "seconds")
    
    # this probably sucks because no KV cache
    print("Tokens/second: ", 1 / delta_avg)


    # quit()
    # FLOP estimate
    # x = jnp.expand_dims(jnp.array(seq).astype(jnp.int32), 0)
    # lowered = jax.jit(fwd).lower(x, gem)

    # compiled = lowered.compile()
    # print("FLOP estimate: ", compiled.cost_analysis())
    # quit()
    
