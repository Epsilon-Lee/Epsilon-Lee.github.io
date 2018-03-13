---
layout: post
title: "Reinforcement Learning Tutorial and Basic Materials: with on emphasis on NLP"
author: Guanlin Li
tag: notes
---

[TOC]

### 1. Basic blog articles

- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/), Andrej Karpathy's blog post on DRL. 
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html), undergraduate now at Google. 

### 2. Serious tutorials

- [Deep Reinforcement Learning, Decision Making, and Control](https://sites.google.com/view/icml17deeprl), ICML 2017 tutorials, Berkeley. 
- [Deep Reinforcement Learning through Policy Optimization](https://media.nips.cc/Conferences/2016/Slides/6198-Slides.pdf), with an emphasis on policy gradient/optimization, Berkeley. 
- [Deep Reinforcement Learning](https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf), ICML 2016 tutorials, David Silver, DeepMind. 

### 3. Basic application in NLP

#### 3.1. RL for structured prediction

- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099), NIPS 2015. 
  - Renaissance of error propagation issue of teacher forcing training for RNN. 
- [Reward augmented maximum likelihood for neural structured prediction](https://arxiv.org/abs/1609.00150), NIPS 2016. 
  - Efficient way to incorporate RL-like effect by label noising and sample reweighing. 
- [Minimum Risk Training for Neural Machine Translation](http://www.aclweb.org/anthology/P16-1159), ACL 2016. 
  - A special case of RL independently developed in statistical machine translation community. 
- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473), AAAI 2017. 
  - Learnable reward function through Discriminator in GAN architecture. 
- [An Actor-Critic Algorithm for Sequence Prediction](https://arxiv.org/abs/1607.07086), ICLR 2016. 
  - Use the classic and strongest RL algorithm for neural s.p. problems.  
- [Self-critical sequence training for image captioning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf), CVPR 2017. 
  - A practical baseline for policy gradient. 
- [Decoding with Value Networks for Neural Machine Translation](https://papers.nips.cc/paper/6622-decoding-with-value-networks-for-neural-machine-translation), NIPS 2017. 
  - Value network (Q-learning) application in NMT decoding. 

#### 3.2. Learning to search (exposure bias)

- [Sequence-to-sequence learning as beam-search optimization](https://arxiv.org/abs/1606.02960), EMNLP 2016. 
  - Max-margin learning2search renaissance. 
  - Explicitly propose exposure bias problem (a nickname for the error propagation problem of MLE during testing or train/test label shift issue). 
- [Professor forcing: a new algorithm for training recurrent networks](https://arxiv.org/abs/1610.09038), NIPS 2016. 
- [SEARNN: Training RNNs with global-local losses](https://openreview.net/forum?id=HkUR_y-RZ), ICLR 2018. 
  - Cost-sensitive training at each RNN time-step, similar to better credit assignment with step-specific reward in RL. 
- [Maximum Margin Reward Networks for Learning from Explicit and Implicit Supervision](http://aclweb.org/anthology/D/D17/D17-1252.pdf), EMNLP 2018. 

#### 3.3. Misc. 

- [Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control](https://arxiv.org/abs/1611.02796), ICML 2017. 

- [Seq2SQL: enerating Structured Queries From Natural Language Using Reinforcement Learning](https://openreview.net/forum?id=Syx6bz-Ab), ICLR 2018, **reject**. 


#### 3.4. RL for Low-resource or transfer/fast adaptation/few-shot/

- [Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](https://arxiv.org/abs/1603.07954), EMNLP 2016 best paper. 
- [Learning how to Active Learn: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1708.02383), EMNLP 2017. 