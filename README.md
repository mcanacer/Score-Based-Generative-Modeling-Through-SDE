# Score-Based Generative Modeling Through SDE from scratch in JAX/FLAX

This repository contains a from-scratch implementation of the paper:

> ** Maximum Likelihood Training of Score-Based Diffusion Models **  
> (https://arxiv.org/abs/2101.09258)



## 🏁 Training SDE

```bash
python train.py configs/cifar10_vpsde.yaml
```

## 🎨 Inference

```bash
python inference.py configs/cifar10_vpsde.yaml
```

## References

Blog https://yang-song.net/

SDE https://github.com/yang-song/score_sde
