import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def ker(x_ref, y_ref, o_ref):
  x,y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


# def add_vec(x, y):
#     return pl.pallas_call(ker,
#                         out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),

#                         )(x, y)

# print(add_vec(jnp.ones(65), jnp.ones(65)*2))

def weave(x):
  odd = x.at[1::2].get()
  even = -x.at[::2].get()
  interleaved = 
  


x = jnp.arange(9)

print(weave(x))