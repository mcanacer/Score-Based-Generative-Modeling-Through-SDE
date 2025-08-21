import sys
import yaml
import os

import torch
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
from torchvision.utils import save_image
from unet import UNet
from sde import VPSDE
from sampling import EulerMaruyamaPredictor, LangevinCorrector
from utils import batch_mul


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def generate_samples(rng, unet, unet_params, sde, predictor, corrector, shape):
    def make_predict_fn(*, apply_fn):
        def predict_fn(params, x, t):
            labels = t * 999
            prediction = apply_fn(params, x, labels)
            return prediction

        return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    devices = jax.local_devices()
    num_devices = len(devices)
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    predict_fn = make_predict_fn(apply_fn=unet.apply)

    params_repl = replicate(unet_params)
    rng, sample_rng = jax.random.split(rng, 2)

    x = jax.random.normal(sample_rng, shape=shape)
    timesteps = jnp.linspace(sde.T, 1e-3, sde.N)

    for t in timesteps:
        rng, sample_rng, step_rng = jax.random.split(rng, 3)
        vec_t = jnp.ones(x.shape[0]) * t  # continuous t
        shard_x = jax.tree_util.tree_map(lambda x: shard(np.array(x)), x)
        shard_t = jax.tree_util.tree_map(lambda x: shard(np.array(x)), vec_t)

        prediction = predict_fn(params_repl, shard_x, shard_t)
        prediction = jax.tree_util.tree_map(lambda x: unshard(np.array(x)), prediction)
        std = sde.marginal_prob(x, vec_t)[1]
        score = batch_mul(-prediction, 1. / std)
        ###
        x = corrector.update_fn(sample_rng, x, score, vec_t)[0]
        ###
        x, x_mean = predictor.update_fn(step_rng, x, score, vec_t)

    x_mean = (x_mean + 1.0) / 2.0
    return jnp.clip(x_mean, 0.0, 1.0)


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    unet_config = config['model']
    sde_config = config['SDE']

    seed = unet_config['seed']
    key = jax.random.PRNGKey(seed)

    unet = UNet(**unet_config['params'])
    vpsde = VPSDE(**sde_config['params'])
    predictor = EulerMaruyamaPredictor(vpsde)
    corrector = LangevinCorrector(vpsde, 0.16, 1)

    checkpoint_path = unet_config['checkpoint_path']
    unet_params = load_checkpoint(checkpoint_path, None)['params']

    x_gen = generate_samples(key, unet, unet_params, vpsde, predictor, corrector, (64, 32, 32, 3))

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/SDE/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
