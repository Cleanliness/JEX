import jax
import jax.numpy as jnp
from functools import partial

def gelu(x, approx=True):
    """
    Gaussian Error Linear Unit
    """
    if approx:
        return jax.nn.gelu(x, approximate=True)
    return 0.5*x*(1 + jax.scipy.special.erf(x / jnp.sqrt(2)))


def swish(x, beta=1.0):
    """
    Swish activation function
    """
    return x*jax.nn.sigmoid(x*beta)


def GLU(x, W1, W2, b1, b2, gate_fn=gelu):
    """
    A generalized gated linear unit

    Apply weights to each time step independently
    Args:
        - x: (B, T, d) input
        - W1: (d_g, d) weight matrix for the gate (gate_proj)
        - W2: (d_g, d) weight matrix for the output (up_proj)
        - b1: (d_g,) bias for the gate
        - b2: (d_g,) bias for the output
    """
    z1 = jnp.einsum('...d, gd->...g', x, W1) + b1
    z2 = jnp.einsum('...d, gd->...g', x, W2) + b2
    return gate_fn(z1)*z2


if __name__ == "__main__":
    glujit = jax.jit(GLU)

    B = 5
    T = 10
    d = 8
    d_g = 4

    x = jnp.arange(B*T*d).reshape((B, T, d))

    # test
    W1 = jax.random.normal(jax.random.PRNGKey(0), (d_g, d))
    W2 = jax.random.normal(jax.random.PRNGKey(0), (d_g, d))
    b1 = jax.random.normal(jax.random.PRNGKey(0), (d_g,))
    b2 = jax.random.normal(jax.random.PRNGKey(0), (d_g,))

    print("====== GeGLU ======")
    r1 = GLU(x, W1, W2, b1, b2)
    r2 = glujit(x, W1, W2, b1, b2)

    print("shape:", r1.shape)
    print("JIT parity:", jnp.allclose(r1, r2))

    print("====== swiglu ======")
    b = 3.0
    swish_m = partial(swish, beta=b)
    swishjit = jax.jit(swish_m)
    r1 = swish_m(x)
    r2 = swishjit(x)

    print("shape:", r1.shape)
    print("JIT parity:", jnp.allclose(r1, r2))

    print("====== correctness ======")
    features = 2
    hidden_dim = 3

    x = jnp.arange(1, 3)[:, None, None]

    # (B, T, features)
    x = jnp.repeat(x, features, axis=-1)

    res = GLU(x, jnp.ones((hidden_dim, features)), jnp.ones((hidden_dim, features)), 0, 0)
    res = res @ jnp.ones((hidden_dim, features))
    b = 3