import jax


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)
