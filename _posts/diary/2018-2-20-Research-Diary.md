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




#### Feb. 22

The flight from Chengdu to Shenzhen, a journey which I felt so delightful for a new departure with the bless of my parents, is a new start of an experience full of challenges and opportunities. 

Here at Tecent AI Lab, with my mentor Leimao and my colleague Xintong, we are working on new algorithm or solution construction paradigm which is friendly to both learning and inference. I will try to doubt the current dominant maximum likelihood estimation (MLE) approach from different viewpoints, namely, maximum entropy, reinforcement learning (first order Q-learning or first order Policy Gradient). Or from a [Bayesian](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) (Stochastic Gradient Langevin Dynamics) or expectation/model averaging viewpoint, from where point estimation (MLE, MXENT or PG) is regarded as (transformed/regularized) mode seeking so that learning with noise and uncertainty is not better guaranteed, which might be the reason that diverse or alternative translations are not easy to sample or search from. I would also like to research on a time-consuming but intuitively well-motivated approach (semi-autoregressive Sequence Modeling/NMT) at the middle spectra with end points as autoregressive and [non-autoregressive](https://arxiv.org/abs/1711.02281) [models](https://arxiv.org/pdf/1802.06901). New solution construction algorithm may reform the beam-search or greedy based inference paradigm towards possibly dynamic programming based decoding algorithms. Another possible research direction is through better experience exploitation through [learned curricula](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf), [self-paced](http://ai.stanford.edu/~pawan/publications/KPK-NIPS2010.pdf) [learning](https://papers.nips.cc/paper/5568-self-paced-learning-with-diversity.pdf). As for training myself to develop better structural bias of the neural architecture and language themselves, I will also research on latent structural bias exploration combined with current state-of-the-art neural machine translation architectures to see if these prior knowledge or latent structures have better exploited. 

##### Some dots, today

Today, I have a happy morning during my flight with reading of Yoshua's great book 'Learning Deep Architectures for AI' [book](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf), it has a clear demonstration of the contrastive divergence (CD) algorithm for learning the restricted Boltzmann machine. The concept of [**mixing**](https://stats.stackexchange.com/questions/223691/what-does-mixing-mean-in-sampling) during MCMC-based sampling is very essential for sampling-based (kind of exploration in reinforcement view) training, and I kind of getting to know the point. Then their is a better intuition explained by him on learning of curricula as a kind of [continuation](http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNd_Numerical%20calculus/Allogower%20Introduction.pdf) [method](http://www.jmlr.org/papers/volume17/gulchere16a/gulchere16a.pdf), which has application in [text classification](https://link.springer.com/content/pdf/10.1023/A:1007692713085.pdf). 

Then I read Alain de Botton's book 'The Art of Travel' which is full of wisdom. [TO-DO tomorrow]. 

In the afternoon, I read through Zhirui's ACL papers and gave some advice on revision. And tomorrow I will meet with him and have a nice chat! 