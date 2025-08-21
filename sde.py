import jax.numpy as jnp
from utils import batch_mul


class VPSDE(object):
    def __init__(self, beta_min, beta_max, N, eps):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_sqrt = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
        self.eps = eps

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * self.beta_min
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
        return mean, std
