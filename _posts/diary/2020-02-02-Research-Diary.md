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



### 3/2/2020

> Monday, it's the new start working day of the year 2020.

#### Demystifying Learning of UNMT

1. <del>Design *warmup* experiments</del>
2. Refine research Qs and some proposed hypothesis

---

**Some conceptual factors**

Here, I summarise several conceptual factors which its theory should be taken a look into so as to get the potential improved or right understanding of the successful learning of UNMT.

- **Autoencoder** in seq2seq framework
  - [paper 1](https://arxiv.org/pdf/1907.04944.pdf) and [paper 2](https://openreview.net/forum?id=ryl3blSFPr)
  - [Denoising autoencoder](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
- **Adversarial loss** ([CycleGAN theory](https://openreview.net/forum?id=B1eWOJHKvB))
- **Co-training theory**
  - [original paper](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)
  - Others (in Zotero)
  - The limit of co-training
  - Better understanding of back-translation *through the lens of* co-training in its 1st iteration.
- **Parameter initializtion evaluation**

---

**Refining research questions**

I think there are at least two questions our community of MT would be interested in:

1. ***Understand the (un-)effectiveness of parameter initialization***
   - `Upper bound`
     - ***Use*** or *develop* the theory of task relatedness of certain self-supervised task and the downstream task, e.g. machine translation.
   - `Lower bound`
     - Develop certain measure, **e.g.** *the number of universal semantic hubs*, so that the measure a). correlates well with empirical experiments; b) is interpretable.
2. ***Understand the (un-)effectiveness of the learning protocol based on certain initialization***
   - `Upper bound`
     - Use **co-training theory** to formulate a (realistic) <u>learning characteristic measure</u> of unsupervised NMT training, which could conceptually help the understanding of supervised training.
   - `Lower bound`
     - the interaction of multiple loss functions: the ablation of loss functions under different initialization methods
       - **Q**: *Can we reproduce the zero BLEU performance of emnlp best paper?*
       - **Q**: How to use *signal-noise ratio* of learning to interpret the success and failure of the learning?
       - *Charaterize* and *compare* the learning of unsupervised and supervised model



#### Miscs

- [Can we identify word senses from deep contextualized word embeddings without supervision?](https://medium.com/@leslie_huang/automatic-extraction-of-word-senses-from-deep-contextualized-word-embeddings-2f09f16e820), a medium post on unsupervised discovery of word sense.
- [The Next Generation of Machine Learning Tools](http://inoryy.com/post/next-gen-ml-tools/), blog post on DL Software advancement, which I try to write a summary blog post about.
- [Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/), blog post from CMU ML feed, happy to learn about `federated learning`.
- `random thought`. To further learn the gist from John Schulman's blog post on *How to do research?*, try to learn from his works on policy optimization in RL and learn how he actually comes up with these ideas and theories and how he conduct persuasive experiments to support his theory or argument?
- Put `<del>xxx</del>` in markdown will have such <del>xxx</del> effect.



### 4/2/2020

> I received response email from Amazon for resetting the coding interview opportunity. Cheers and face it, bud!



#### The `TODO`s of a day

1. Work on the `Demystifying Learning of UNMT` project.
2. Prepare the coding interview on next Friday.
3. Practice guitar and TABS writing in guitar pro 7.5.

