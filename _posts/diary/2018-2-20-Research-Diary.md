---
layout: post
title: "Research Diary from Feb. 2 - to Feb. 25"
author: Guanlin Li
tag: diary
---

#### Feb. 20

- [Tensor Comprehension](https://arxiv.org/abs/1802.04730). DL frameworks explores balance of usability and expressiveness. They operate on DAG of computational operators, wrapping high-performance libraries such as CUDNN for NVIDIA GPUs and NNPACK for various CPUs, and automatic memory allocation, synchronization, distribution. The **drawbacks** of existing frameworks are: newly-designed operations could not fit quickly into frameworks with high-performance guarantee. [The research blog for TC](https://research.fb.com/announcing-tensor-comprehensions/). 
  - [This](https://facebookresearch.github.io/TensorComprehensions/) is the online doc for TC. 
  - TC is a notation based on generalized Einstein notation for computing on multi-dimensional arrays. TC greatly simplifies ML framework implementations by providing a concise and powerful syntax which can be efficiently translated to high-performance computation kernels, automatically. 
- [A blog post for dehyping reinforcement learning](https://www.alexirpan.com/2018/02/14/rl-hard.html), because it is not the panacea with 70% confidence. 
- Some papers waiting for a read. 
  - [Make the Minority Great Again: First order regret bound](https://arxiv.org/pdf/1802.03386.pdf), COLT 2018, online learning. 
  - Reinforcement learning related. 
    - [Reinforcement Learning from Imperfect Demonstrations](https://arxiv.org/pdf/1802.05313.pdf), ICML 2018 submitted. 
    - [Mean-field multi-agent reinforcement learning](https://arxiv.org/pdf/1802.05438.pdf), ICML 2018 submitted. 
  - [Universal Neural Machine Translation for Extremely Low Resource Languages](https://arxiv.org/pdf/1802.05368.pdf), NAACL 2018. 
  - [SparseMAP: Differentiable Sparse Structured Inference](https://arxiv.org/pdf/1802.04223.pdf), structured prediction with inference and explanation power. 
  - [Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction](https://arxiv.org/pdf/1802.05451.pdf), submitted to ICML 2018. 
  - Representation learning.
    - ["Dependency Bottleneck" in Auto-encoding Architectures: An empirical study](https://arxiv.org/pdf/1802.05408.pdf). ICLR 2018. 
  - Meta-learning.
    - [Learning to Learn without Labels](https://openreview.net/forum?id=ByoT9Fkvz), ICLR 2018.
  - [Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916), Alex Smola. 

