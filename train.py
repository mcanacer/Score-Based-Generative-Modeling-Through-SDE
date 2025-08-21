import sys
import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from unet import UNet
import sde
from utils import batch_mul

import numpy as np


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def make_update_fn(*, apply_fn, optimizer, sde, ema_decay):
    def update_fn(params, opt_state, images, rng, ema_params):
        def loss_fn(params):
            time_rng, sample_rng = jax.random.split(rng, 2)

            t = jax.random.uniform(
              time_rng,
              minval=float(sde.eps),
              maxval=sde.T,
              shape=(images.shape[0],)
              )

            mean, std = sde.marginal_prob(images, t)
            z = jax.random.normal(sample_rng, shape=images.shape)

            noisy_images = mean + batch_mul(std, z)

            labels = t * 999
            predicted_noise = apply_fn(params, noisy_images, labels)

            g2 = sde.sde(jnp.zeros_like(images), t)[1] ** 2
            score = batch_mul(-predicted_noise, 1. / std)

            loss = jnp.mean((batch_mul(score, std) + z) ** 2)

            return loss

        loss, grad = jax.value_and_grad(loss_fn)(params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_ema_params, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    unet_config = config['model']
    sde_config = config['SDE']
    dataset_params = config['dataset_params']
    wandb_config = config['wandb']

    seed = unet_config['seed']

    transform = transforms.Compose([
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.RandomHorizontalFlip(0.5),
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    train_dataset = torchvision.datasets.CIFAR10('.', train=True, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_params['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers'],
        drop_last=True,
    )

    score_model = UNet(**unet_config['params'])
    vpsde = sde.VPSDE(**sde_config['params'])

    optimizer = optax.chain(
        optax.adam(
            learning_rate=float(unet_config['learning_rate']),
            b1=unet_config['b1']
        )
    )

    epochs = unet_config['epochs']

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )

    checkpoint_path = unet_config['checkpoint_path']

    key = jax.random.PRNGKey(seed)
    key, sub_key, other_key = jax.random.split(key, 3)

    inputs = next(iter(train_loader))[0]

    fake_timesteps = jax.random.uniform(other_key, minval=float(vpsde.eps), maxval=vpsde.T, shape=(inputs.shape[0],))
    params = score_model.init(sub_key, inputs, fake_timesteps)

    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_decay = unet_config['ema_decay']
    ema_params = params
    ema_params_repl = replicate(ema_params)

    update_fn = make_update_fn(
        apply_fn=score_model.apply,
        optimizer=optimizer,
        sde=vpsde,
        ema_decay=ema_decay,
    )

    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)

    del params
    del opt_state

    num_devices = jax.local_device_count()

    state_template = {
        "params": unreplicate(params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        "epoch": 0,
        "rng": key,
    }

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    if loaded_state is not None:
        print("Resuming from checkpoint...")
        params_repl = replicate(loaded_state["params"])
        opt_state_repl = replicate(loaded_state["opt_state"])
        ema_params_repl = replicate(loaded_state["ema_params"])
        key = loaded_state["rng"]
        start_epoch = loaded_state["epoch"] + 1
    else:
        start_epoch = 0

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for step, (images, _) in enumerate(train_loader):
            key, step_rng = jax.random.split(key, 2)

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            rng_shard = jax.random.split(step_rng, num_devices)

            (
                params_repl,
                opt_state_repl,
                ema_params_repl,
                loss,
            ) = update_fn(
                params_repl,
                opt_state_repl,
                images,
                rng_shard,
                ema_params_repl,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        checkpoint_state = {
            "params": unreplicate(params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            "epoch": epoch,
            "rng": key,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
