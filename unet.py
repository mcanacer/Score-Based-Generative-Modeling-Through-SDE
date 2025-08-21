import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import List

import math


class Upsample(nn.Module):
    filters: int
    factor: int = 2
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x):
        N, H, W, C = x.shape
        x = jax.image.resize(x, shape=(N, H * self.factor, W * self.factor, C), method=self.method)
        x = nn.Conv(self.filters, kernel_size=(3, 3))(x)
        return x


class Downsample(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, kernel_size=(3, 3), strides=2)(x)
        return x


class ResidualBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, timestep):
        residual = x
        x = nn.GroupNorm(num_groups=32, dtype=jnp.float32)(x)
        x = jax.nn.swish(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3))(x)  # [N, H, W, C]

        timestep = jax.nn.swish(timestep)  # [N, C]
        timestep = nn.Dense(self.filters)(timestep)  # [N, C]
        timestep = jnp.expand_dims(timestep, axis=(1, 2))  # [N, 1, 1, C]

        x = x + timestep  # [N, H, W, C]
        x = nn.GroupNorm(num_groups=32, dtype=jnp.float32)(x)

        x = jax.nn.swish(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3))(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(self.filters, kernel_size=(1, 1))(residual)

        return x + residual


class Attention(nn.Module):
    n_heads: int = 1

    @nn.compact
    def __call__(self, inputs):
        N, H, W, C = inputs.shape
        x = inputs.reshape(N, H * W, C)

        attention_mask = nn.make_attention_mask(
            jnp.ones((N, H * W), dtype=jnp.int32),
            jnp.ones((N, H * W), dtype=jnp.int32)
        )

        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=C,
            out_features=C,
            dropout_rate=0.0,
            deterministic=True
        )(x, attention_mask)

        x = x.reshape(N, H, W, C)
        return nn.GroupNorm(num_groups=32, dtype=jnp.float32)(x)


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, timesteps):
        half_dim = self.dim // 2
        emb = -math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * emb)
        emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class UNet(nn.Module):
    channel_multiplier: List[int]
    attn_strides: List[int]
    channel: int = 128
    n_res_block: int = 2
    attn_heads: int = 1

    @nn.compact
    def __call__(self, images, timesteps):
        t = SinusoidalPosEmb(self.channel)(timesteps)
        t = nn.Dense(self.channel * 4)(t)
        t = jax.nn.swish(t)
        t = nn.Dense(self.channel * 4)(t)
        time_embeds = t

        feats = []

        x = nn.Conv(self.channel, kernel_size=(3, 3))(images)
        n_blocks = len(self.channel_multiplier)

        # Encoder
        for i in range(n_blocks):
            for _ in range(self.n_res_block):
                channel_mul = self.channel * self.channel_multiplier[i]
                x = ResidualBlock(filters=channel_mul)(x, time_embeds)

                if 2 ** i in self.attn_strides:
                    x = Attention(n_heads=self.attn_heads)(x)

            if i != n_blocks - 1:
                feats.append(x)
                x = Downsample(filters=x.shape[-1])(x)

        # Bottleneck
        for _ in range(self.n_res_block):
            x = ResidualBlock(filters=x.shape[-1])(x, time_embeds)

        # Decoder
        for i in reversed(range(n_blocks - 1)):
            x = Upsample(filters=x.shape[-1])(x)
            skip = feats.pop()
            x = jnp.concatenate([x, skip], axis=-1)

            for _ in range(self.n_res_block):
                channel_mul = self.channel * self.channel_multiplier[i]
                x = ResidualBlock(filters=channel_mul)(x, time_embeds)

                if 2 ** i in self.attn_strides:
                    x = Attention(n_heads=self.attn_heads)(x)

        x = nn.GroupNorm(num_groups=32)(x)
        x = jax.nn.swish(x)
        x = nn.Conv(images.shape[-1], kernel_size=(3, 3))(x)

        return x


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    sub_key = jax.random.PRNGKey(1)
    model = UNet([1, 1, 2, 2, 4, 4], [16])
    fake_timesteps = jax.random.randint(sub_key, minval=0, maxval=1000, shape=(2,))
    fake_images = jnp.ones((2, 128, 128, 3), dtype=jnp.float32)
    params = model.init(key, fake_images, fake_timesteps)
