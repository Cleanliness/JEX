import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

from typing import NamedTuple

def ker(x_ref, y_ref, o_ref):
  x,y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


# def add_vec(x, y):
#     return pl.pallas_call(ker,
#                         out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),

#                         )(x, y)

# print(add_vec(jnp.ones(65), jnp.ones(65)*2))

# test doing a scan over layered params


class nested_params(NamedTuple):
  e: jnp.ndarray
  f: jnp.ndarray

class Params(NamedTuple):
  a: jnp.ndarray
  b: jnp.ndarray
  c: jnp.ndarray
  d: nested_params

nested_params = nested_params(
  e=jnp.ones((3, 10)),
  f=jnp.ones((3, 10))
)

test_params = Params(
  a=jnp.ones((3, 10)),
  b=jnp.ones((3, 10)),
  c=jnp.ones((3, 10)), 
  d=nested_params
)

def test_scan(x, params):
  a,b,c,d = params
  e,f = d
  return a+b+c+e+f+x, ()

print(jax.lax.scan(test_scan, jnp.zeros(10), test_params))
