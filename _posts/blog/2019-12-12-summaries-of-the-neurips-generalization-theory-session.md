---
layout: post
title: "Summaries of the NeurIPS 2019 Track 1 Session 1"
author: Guanlin Li
tag: blog
---

This track is about theoretical and empirical generalization analyses of deep learning, which is very to my interesting, since I am absorbed by the mystery of the realistic generalization of neural models such as the generalization of a sequence-to-sequence model and so on.

##### Uniform convergence may be unable to explain generalization in deep learning

> The understanding new direction paper of this year.

Their high-level motivation is the "overuse" of uniform convergence theory to derive generalization bounds for deep neural networks (heavily over-parameterized).

$$\text{test_error} - \text{train_error} \leq \text{bound}$$

They argue that:

> "This active, on-going direction of research - of using the learning-theoretic tool of uniform convergence to solve the generalization puzzle - may not lead us to the answer."

They have two main findings in their paper:

1. bounds grow with training set size;
2. provable failure of uniform convergence;

**Background on u.c. bounds**

*Conventional* uniform convergence (u.c.) bounds: e.g. the VC dimension has the following form:

$$\text{generalization_gap} \leq O(\sqrt{\frac{\text{representational complexity of whole hypotheses class}}{\text{training set size}}})$$

however, for NNs, the numerator "representational complexity of whole hypotheses class" can be related to millions of parameters, which makes this upper bound too large. So recent efforts proposed to refine the above bound by:

$$\text{generalizat_gap} \leq O(\sqrt{\frac{\text{representation complexity of "relevant" subset of hypothesis class}}{\text{training set size}}})$$

That is instead of measuring the complexity of the whole hypo. class, they turn to measure e.g. parameter's distance from initialization, spectral norm, $$L_{2, 1}$$ norm etc. All these works can be categorized into the following four:

- Rademacher complexity;
- Covering numbers;
- PAC-Bayes;
- Compression;

Unfortunately, these bounds are:

- either large/grow with param. count (unlike actual generalization gap);
- or small but don't hold on original network.

So the authors want to take a step-back to understand what those bounds capture in realistic situation, in light of their limitations. This leads to their first finding:

- bounds grow with training set size;

