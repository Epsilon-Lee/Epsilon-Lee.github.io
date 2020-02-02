---
layout: post
title: "Research Diary from 2/2/2020 - 2/8/2020"
author: Guanlin Li
tag: diary
---

**Table of Content**

* [2/2/2020](#2/2/2020)
* [3/2/2020](#3/2/2020)
* [4/2/2020](#3/2/2020)
* [5/2/2020](#3/2/2020)
* [6/2/2020](#3/2/2020)
* [7/2/2020](#3/2/2020)
* [8/2/2020](#3/2/2020)

### 2/2/2020

> Sunday, actually should go back to Shenzhen, but...

#### Demystify Learning of UNMT

1. Reread research questions on `qqdocs`
2. Reread paper draft on `overleaf`
3. Based on that reread critical original papers to design *warmup* experiments

---

**Highlight of reserch Qs**

In `overleaf`, the most high-level research question is:

> **Q0**. How the proposed training protocol successfully train a sequence-to-sequence model that achieves decent translation performance?

In `qqdocs`, the most high-level research question is:

> **Q1**. How the <u>**dual** self-training paradigm</u> with other tricks like seq2seq self-supervised pertaining (or other loss functions, e.g. denoising, adversarial) can essentially determine the successful learning of the seq2seq translation model?
>
> > **Definition (Dual self-training)**
> >
> > Bootstrap the model itself to generate $$\hat{x}^{l_2}$$ from $$x^{l_1} \in \mathcal{D}^{l_1}$$, and supervised learning from $$(\hat{x}^{l_2} \Rightarrow x^{l_1})$$.

**Q1 is more specific than Q0, since it selects or disentangles several specific elements of the learning of UNMT.**

- The losses
- The pretraining effects

---

In `overleaf`,  I have indicated two overlapping directions on obtaining (principled) understanding of the above **Q0** and **Q1**.

> **Basic principles** which could be measured through quantities, and correlates well with healthy learning. For examples:
>
> - Gradual distribution matching (hierarchically from lexical to phrasal and finally sentential semantics)
>   - *What apporatus facilitates the compositional learning effect?*
> - Latent anchor semantic hubs, that prevent learning from catastrophic failures
>   - Universal semantics and how to measure it?
>
> **Learning dynamics** which could directly visualize or reflect some failure modes or noise-level of training.
>
> - ***Interaction*** of the training losses, with the actual bilingual likelihood or perplexity
>   - acc. `qqdocs`, Lemao proposed to compare losses influence to LCA of  bilingual likelihood to multiple (simultaneous) objectives.
>   - Say the bilingual likelihood loss is $$l_{bi}$$, the other four losses are $$l_{d}^{l_1}, l_{d}^{l_2}, l_{bt}^{l_1}, l_{bt}^{l_2}$$, can we use loss correlation instead of the concept of allocation?
>     - In terms of allocation, since we don't know the function that links say $$l_{d}^{l_1}$$ to $$l_{bi}$$, so the computation of original LCA cannot transfer to this situation directly.
> - **Critical Model Components Identification** in terms of the original LCA. 
>   - Since the dual parameterization of UNMT model, how does it compare to supervised NMT model with or without such parameterization?

In `qqdocs`, we also detailed on certain loss dynamics analysis settings.

> - **Different learning phrases**, in terms of the implicit supervised bilingual loss.
>   - which is a charaterization of the training of UNMT.
>   - Can training be divided into several evident stages:
>     - Noise-resistant phase (slow learning, warm-up phases)
>     - Quick fitting phase
>     - Convergence phase
>   - `TODO`: refer to existing papers for possible answers.
> - **Information-theoretic interpretation**
>   - Mutual information measure
>   - Information bottleneck (???), the unsupervised version.

#### Miscs

- [New blog post by Gregory Gundersen: Can Linear Models Overfit](http://gregorygundersen.com/blog/2020/01/31/linear-overfitting/).