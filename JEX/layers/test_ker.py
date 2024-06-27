import jax
import jax.numpy as jnp

@jax.jit
def foo(x):
  res = 0
  for i, j in enumerate(x):
    x[i] = jnp.sum(j)
  return x

test = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]


print(foo(test))
