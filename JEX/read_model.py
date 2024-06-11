from safetensors.flax import load_file
import jax
import jax.numpy as jnp

MODEL_DIR = r"/media/roy/TOSHIBA EXT/models/gemma-2b/"

p1 = "model-00001-of-00002.safetensors"
p2 = "model-00002-of-00002.safetensors"

def load_model():
    model_params = load_file(MODEL_DIR + p1)
    model_params2 = load_file(MODEL_DIR + p2)

    # merge dicts
    model_params.update(model_params2)
    return model_params


adfasdf = 3

