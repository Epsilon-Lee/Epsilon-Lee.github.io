---
layout: post
title: "On Learning Under Dataset Noise"
author: Guanlin Li
tag: blog
---

I am extremely curious about how *data* or *experience* drives the model to learn imperfect but useful prediction rules. One of the aspect of such learning process is the <u>noisy experience in the data</u> from which the model is going to learn. At the last day of the second decade of the 21st century. I would like to summarize some papers that I have encountered along my daily browsing.

> Note that, learning under noise is highly correlated with topics like curriculum learning, data augmentation, *data selection* and active learning, which may or may not be covered in this post, based on which I hope one day I would write something about.

I want to do this summary due to the very paper named:

- [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/pdf/1911.00068.pdf), which has submitted to *AISTATS 2020*.

This is a method paper for empirical improvement. Their basic ideas are:

1. Dataset pruning
2. Examples ranking
3. Confidence-weighed training

which once done properly, I think, is the best practice of learning under noise.

One thing that really interests me is their so-called **model-agnostic dataset uncertainty estimation** method.



**Learning under noise**

- [Understanding and Utilizing Deep Neural Networks
  Trained with Noisy Labels](http://proceedings.mlr.press/v97/chen19g/chen19g.pdf), ICML 2019.
- [Unsupervised Label Noise Modeling and Loss Correction](http://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf), ICML 2019.
- [Learning with Bad Training Data via Iterative Trimmed Loss Minimization](http://proceedings.mlr.press/v97/shen19e/shen19e.pdf), ICML 2019.

**Uncertainty estimate**

- [On Discriminative Learning of Prediction Uncertainty](http://proceedings.mlr.press/v97/franc19a/franc19a.pdf), ICML 2019.

**Data selection**

- [Learning and Data Selection in Big Datasets](http://proceedings.mlr.press/v97/ghadikolaei19a/ghadikolaei19a.pdf), ICML 2019.

- [Metric-Optimized Example Weights](http://proceedings.mlr.press/v97/zhao19b/zhao19b.pdf), ICML 2019.

