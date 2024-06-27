import jax
import jax.ad_checkpoint
import jax.numpy as jnp

from JEX import gemma, tokenizer
import os

# performance flags

def simple_sample(logits, rng, temp=1, topk=50):
    """
    A basic sampler

    Args:
        logits: Model logits (B, n_vocab)
        rng: jax prng key
    """

    # TODO: pair logits with token ids
    logits = jax.lax.top_k(logits, topk)
    logits /= temp

    probs = jax.nn.softmax(logits)
    pass



# A simple example of inference
if __name__ == "__main__":
    MAX_TOKENS = 60

    print("loading model")
    tk = tokenizer.load_tokenizer(r"/media/roy/TOSHIBA EXT/models/gemma-2b/tokenizer.model")
    eos_tok = tk.eos_id()    

    model = gemma.load_gemma_model()
    print("model loaded")

    fwd = gemma.construct_gemma_forward(model)
    fwd = jax.jit(fwd)

    usr_in = []
    while True:
        print("IN:")
        while True:
            curr_ln = input()

            if curr_ln == "quit":
                quit()

            elif curr_ln == "":
                break

            usr_in.append(curr_ln)
        

        print("Completion:")
        seq = tokenizer.encode_str("\n".join(usr_in), tk)
        t = None
        kv = None
        # generation loop
        for _ in range(MAX_TOKENS):
            x = jnp.expand_dims(jnp.array(seq).astype(jnp.int32), 0)
            logits, kv = fwd(x, model, kv, t=t)
            res = jnp.argmax(logits, axis=-1).tolist()

            t = len(seq) + 1
            seq = res

            if res[-1] == eos_tok:
                break
            print(f"{tokenizer.decode_tk(res, tk)}", end='')

        usr_in = []
        curr_ln = ""

