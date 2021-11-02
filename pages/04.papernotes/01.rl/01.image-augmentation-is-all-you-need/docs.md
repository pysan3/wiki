---
title: Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels
taxonomy:
  category: docs
---

## Refs

- [ICLR Session](https://iclr.cc/virtual/2021/poster/3188)
- [OpenReview](https://openreview.net/forum?id=GY6-6sTvGaf)
- [Download PDF](not ready)
- [Implementation (Github)](https://github.com/denisyarats/drq)

### Bibtex

```text
@inproceedings{
  yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```

## Overview

- An augmentation technique that can be applied to standard model-free RL algorithms.
- Enable robust learning from pixels without pre-training.
- Named: **DrQ**(Data-regularized Q)
  - Can be combined with any mode-free RL algorithm
- Leverages input perturbations(摂動).
- Regularizes the value function and policy.

### Experiments

- DeepMind control suite
- Atari 100k benchmark

### Contributions

- data augmentation improves performance of model-free RL algorithms for images
- two mechanisms for regularizing the value function for MDP(マルコフ決定過程) structure
- new state-of-the-art performance on standard DeepMind, Atari 100k etc

## Related Works

- model-based methods
  - get compact latent representation of high-dimensional observation
- model-free methods
  - to learn the latent representation indirectly
    - optimize the RL objective
    - employing auxiliary loss

**Our methods is complementary to the `model-free methods`**

## Background

省略

## Our Method

### Optimality Invariant Image Transformations for Q Function

- To regularize the value function through transformation of input

define: $f:\mathcal{S}\times\mathcal{T}\rightarrow\mathcal{S}$, $\mathcal{T}:f$に渡すパラメータの取りうる値(詳しくは次の章で)

Q-value is calcutated with:

$$
Q(s,a)=Q(f(s;v),a)\textrm{ for all }s\in \mathcal{S},a\in\mathcal{A}\textrm{ and }v\in\mathcal{T}
$$

Therefore, instead of using a single sample: $s^*\sim\mu(\cdot)$, $a^*\sim\pi(\cdot|s^*)$, which gives

$$
\begin{align*}
  \mathbb{E}_{s\sim\mu(\cdot),a\sim\pi(\cdot|s)}&[Q(s,a)]\\
  \approx&Q(s^*,a^*)
\end{align*}
$$

Use,

$$
\begin{align*}
  \mathbb{E}_{s\sim\mu(\cdot),a\sim\pi(\cdot|s)}&[Q(s,a)]\\
  \approx&\frac{1}{K}\sum^K_{k=1}Q(f(s^*;v_k),a_k)\\
  &\textrm{ where }v_k\in\mathcal{T}\textrm{ and }a_k\sim\pi(\cdot|f(s^*;v_k))
\end{align*}
$$

#### Note:

Q 関数を計算するときに、ある state:$s^*$で、そのまま$Q(s^*,a^*)$を計算せずに、$f(s;v_k)$で$s^*$に**似た**state を$K$個生成して計算した Q 値の平均を使う。

To regularize the Q function, target value is calcutated by

$$
y_i=r_i+\gamma\frac{1}{K}\sum^K_{k=1}Q_\theta(f(s_i';v_{i,k}'),a_{i,k}')
$$

then, Q function is updated using,

$$
\theta\leftarrow\theta-\lambda_\theta\nabla_\theta\frac{1}{N}\sum^N_{i=1}(Q_\theta(f(s_i;v_i),a_i)-y_i)^2
$$

Note that, multiple $f(\cdot)$ can be used to regularize Q function, **if $v_{i,m}$ and $v_{i,k}'$ are independent**.

$$
\theta\leftarrow\theta-\lambda_\theta\nabla_\theta\frac{1}{NM}\sum^{N,M}_{i=1,m=1}(Q_\theta(f(s_i;v_{i,m}),a_i)-y_i)^2
$$

### Practical Instantiation of Optimality Invariant Image Transformation

For this paper, we implemented the $f(\cdot)$ with simple shifts of the image.

which shifts the image by $\pm 4$ pixels, horizontally / vertically.

### Our Approach: Data-regularized Q (**DrQ**)

Union of all mechanisms above:

1. transformations of the input image
1. averaging the Q target over $K$ image transformations
1. averaging the Q function itself over $M$ image transformations

## Experiment

- Env
  - DeepMind control suite
- Model
  - unmodified SAC
  - NatureDQN
  - Dreamer
  - ...
