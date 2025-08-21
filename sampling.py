import jax
import jax.numpy as jnp
from utils import batch_mul
from scipy import integrate

import numyp as np


class EulerMaruyamaPredictor(object):

    def __init__(self, sde):
        self.sde = sde

    def update_fn(self, rng, x, score, t):
        dt = -1. / self.sde.N
        z = jax.random.normal(rng, shape=x.shape)
        drift, diffusion = self.sde.sde(x, t)
        x_mean = x + (drift - batch_mul(diffusion ** 2, score)) * dt
        x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
        return x, x_mean


class LangevinCorrector(object):

    def __init__(self, sde, snr, n_steps):
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, rng, x, score, t):
        timestep = (t * (self.sde.N - 1) / self.sde.T).astype(jnp.int32)
        alpha = self.sde.alphas[timestep]

        noise = jax.random.normal(rng, shape=x.shape)
        grad = score
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        step_size = 2 * (self.snr * noise_norm / grad_norm) ** 2 * alpha
        x_mean = x + batch_mul(step_size, grad)
        x = x_mean + batch_mul(jnp.sqrt(2 * step_size), noise)
        return x, x_mean


def ode_sampler(rng, apply_fn, params, sde, atol=1e-5, rtol=1e-5, eps=1e-3):
    rng, sample_rng = jax.random.split(rng, 2)
    z = jax.random.normal(sample_rng, shape=(8, 1, 32, 32, 3))
    init_x = z

    shape = init_x.shape

    def ode_func(time_steps, sample):
        """A wrapper of the score-based model for use by the ODE solver."""
        time_steps = np.ones((8, 1)) * time_steps
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(8, 1, 32, 32, 3)
        time_steps = jnp.asarray(time_steps).reshape(8, 1)
        pred = apply_fn(params, sample, time_steps)
        std = sde.marginal_prob(sample, time_steps)[1]
        score = batch_mul(-pred, 1. / std)
        drift, diffusion = sde.sde(sample, time_steps)
        drift = drift - batch_mul(0.5 * (diffusion ** 2), score)
        return drift.reshape(-1)

    res = integrate.solve_ivp(ode_func, (1., eps), np.asarray(init_x).reshape(-1),
                            rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = jnp.asarray(res.y[:, -1]).reshape(shape)

    return x
