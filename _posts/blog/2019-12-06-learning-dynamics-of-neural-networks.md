---
layout: post
title: "Learning Dynamics of Neural Networks: A brief review"
author: Guanlin Li
tag: blog
---

[TOC]

> This review is targeted on our recent project of analyzing learning dynamics for Neural Machine Translation (NMT) models. When I first try to grasp a taste of the related literature in **understanding learning dynamics of NN**, I find it is a very new directions in current pursuit for the theory of deep learning. So it is necessary for a brief review of bunch of works for this topic.

#### A historical view

Along the timeline, [1] might be the first work that explicitly names out this new direction. [1] proposes a representation comparison method called Singular Vector Canonical Correlation Analysis (SV-CCA) for efficiently comparing *representation* learned through *different hidden layers* from *different training moment*. They find that throughout training, networks converge to final representations from bottom up as shown in Figure 4 of the paper. We call this kind of analysis <u>"representation dynamics"</u>.

> Actually, the focus of study in [1] is representation similarity, their proposed SV-CCA techniques is a way to compute correlation between two groups of representations. This paper points to a direction called "intrinsic dimension" of the hidden representation, which also has connection to the model sparsity/compression observation later on [2].

**Application of SV-CCA**

Based on the SV-CCA technique, [3] starts to look into NLP models like neural language models and how their evolved representation along training corresponding to certain linguistic phenomemon. Besides SV-CCA, [3] also present a measure of representation convergence through *concentration*, which is defined on a word in a context $c(x) = \frac{\vert\vert x \vert\vert_2}{\vert\vert x \vert\vert_1}$, their motivation for this quantity is:

> "if a neural network relies heavily on a small number of cells in an antivation pattern, the activation is very concentrated. Likewise, a cencentrated gradient is mainly modifying a few specific pathways."

Here the $l_1$ norm

---

#### References

[1]. [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability.pdf), NeurIPS 2017.

[2]. [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635), ICLR 2019.

[3]. [Language Models Learn POS First](https://www.aclweb.org/anthology/W18-5438.pdf), Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP.

[4]. [Understanding Learning Dynamics Of Language Models with SVCCA](https://www.aclweb.org/anthology/N19-1329.pdf), NAACL 2019.

---









