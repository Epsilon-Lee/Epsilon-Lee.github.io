---
layout: post
title: "A Critical Review of ACL 2019 Best Long Paper"
author: Guanlin Li
tag: blog
visible: 0
---

[TOC]

### Causality

The cause that motivates me to write this blog post is due to my **re-read** of this year's ACL's best long paper [Bridging the Gap between Training and Inference for Neural Machine Translation](https://www.aclweb.org/anthology/P19-1426). When I know it has received the honor, I start to re-read and try to find out the conceptual advance (which I haven't perceived during my first skim) the paper has made towards this **issue** of *exposure bias* or *train/test mis-match* which has already been **rich** in its literature.

I used to work on this issue when I encountered the conceptual fascination of the problem via Sam Wiseman's [Beam Search Optimization](https://aclweb.org/anthology/D16-1137) paper and then Mohammad Norouzi's mind-sweeping [RAML](https://arxiv.org/pdf/1609.00150.pdf) paper, and further appealed with earlier works like Samy Bengio's [Scheduled Sampling](https://arxiv.org/pdf/1506.03099.pdf) and Marc' Aurelio Ranzato's [Sequence-level Training](https://arxiv.org/pdf/1511.06732.pdf).

At that time, my first grasp of the solution of the troubling *teacher forcing* or *maximum likelihood training (MLE)* of sequence-to-sequence model was:

> Using statistics (word rank, probability [nucleus](https://arxiv.org/pdf/1904.09751.pdf)) from the model's probability scores $$P_\theta(\cdot \vert \hat{y}_{<t}, x)$$ to construct certain sample $$\hat{y}$$ with relatively high model score, while evaluating $\hat{y}$ against $$y^*$$ to get sequence-level supervision.

This conceptual framework is known as the Learning-to-Search (L2S) regime and can be at least, to my knowledge, traced back to Hal Daume's Search-based optimization formulation of Structured Prediction (SP) problems ([conf. paper](https://arxiv.org/pdf/0907.0809.pdf), and [journal paper](https://link.springer.com/content/pdf/10.1007/s10994-009-5106-x.pdf)), which he has realised at that time SP's [connection](http://users.umiacs.umd.edu/~hal/docs/daume05search.pdf) to Reinforcement Learning (RL) but later tried to clarify their [difference](https://nlpers.blogspot.com/2017/04/structured-prediction-is-not-rl.html). This regime is called L2S because the learning is based on inference-then-optimize paradigm, so that training just matches the test.

Due to SP's connection to RL, a bunch of papers including [Sequence-level Training](https://arxiv.org/pdf/1511.06732.pdf), [Actor-Critic (AC)](https://arxiv.org/pdf/1607.07086.pdf), [GAN-like](http://papers.nips.cc/paper/6099-professor-forcing-a-new-algorithm-for-training-recurrent-networks.pdf) [inverse](https://arxiv.org/pdf/1704.06933.pdf) [RL](https://www.aclweb.org/anthology/N18-1122) works are then proposed by different groups. The main idea is to:

> Closing the gap between training and test by optimising the inference or decoding process during training, that is to optimise according to certain *on-policy reward* which can be metric-relevant (sentence-level, or decomposed to be step-wise) or even learned (by Discriminator in GAN or Critic in AC-based RL).

Actually, most of such algorithms or models can be reformulated under the following mathematical form:

$$\nabla_\theta = Q(\hat{y}_t; \hat{y}, x, y^*) \cdot \nabla P_\theta(\hat{y}_t \vert x, \hat{y}_{<t})$$

