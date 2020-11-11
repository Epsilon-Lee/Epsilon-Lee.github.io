---
layout: post
title: "Research Diary from 3/30/2020 - 4/3/2020"
author: Guanlin Li
tag: diary
---



- toc
{:toc}


### Mar. 30

> It's rainy outside.

#### Today's `TODO` list

1. Draw moment, cumulative, loss contribution plots of DAE update and BT update;
2. Write code for arbitrary, fixed, DAE batch;
3. Write code for using expected model score for unsupervised attribution;
   - Problem with truncated mean of BLEU, but how about un-truncated mean of model score? $\Rightarrow$ Do experiment with un-truncated BLEU and measure top-k $\tau$.



### Mar. 31

> Recently, misty weather in Shenzheng.

$\text{NLL}(x) = \frac{1}{\vert x \vert} - \sum_i \log P(x_i \vert x_{<i})$

$\text{PPL}(x) = 2^{\text{NLL}(x)}$

$\mathcal{L} = \frac{1}{N} \sum_n \log P(x^n; \theta) = \frac{1}{N} \sum_n \sum_i \log P(x^n_i \vert x^n_{<i}; \theta)$

假设有K个词，平均每个句子，那么：$2^{\frac{100}{k}} \approx 116$，参看表格2的RNNLM的Test NLL、PPL，能够估算出Penn TreeBank的评价句子长度为：$\log_2 116 = 100 / k$，那么：$6 < 100/k < 7$，那$k$才16-17之间，是不是短了一些？

- [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue), TACL 2016.

#### Today's `TODO` list

1. Enhance the conclusion that:
   - ***DAE update during standard training protocol plays very tiny role of minimizing the latent bitext bilingual loss compared to BT update.***
   - How to? Under the following four loss evaluation settings.
     - Evaluate on DAE batch and switch update order
       - Both are running to 10000 updates;
     - Evaluate on BT batch and switch update order
     - Evaluate on a random batch other than the DAE/BT batch and switch update order
     - Evaluate on a fixed batch and switch update order
       - i) is running;
2. etc.

#### 和昕的讨论

- loss contribution这个量，是否和learning rate、learning rate schedule很相关呢？以至于统计他们各自的contribution会很随机：例如对于学习率是否很敏感，换一个学习率策略结果就不一样了;
- 如果将DAE的联合损失替换成一个解码器上的单语，AR的损失或者编码器解码器各自都有的AR损失会如何呢？



### Apr. 2

> It's sunny then cloudy outside.

#### Today's `TODO` list

1. Simply use DAE training and say if it reduces bilingual loss, how and why?





### Apr. 3

> Sunny outside now, might turn to cloudy.

#### Today's `TODO` list

1. Bought a 30-note music box, punching tools and blank music score papers. `(finished)`
2. Write code for running on the same batch for different runs and gather the loss degrade statistics;



### Apr. 8

> It's Wednesday, it even gets Sunny outside.



#### Miscs.

- [Nice blog](https://mostafadehghani.com/2019/05/05/universal-transformers/#discussion) about *Universal Transformer*.
- The discussion about making Transformer base model work on PTB dataet for language modeling.
  - [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf), ACL 2019.
  - [Language Models with Transformers](https://arxiv.org/pdf/1904.09408.pdf), Alex. Smola.
  - [The Importance of Being Recurrent for Modeling Hierarchical Structure](https://www.aclweb.org/anthology/D18-1503.pdf), EMNLP 2018.
- A several papers on multi-task learning, since I want to get more knowledge about how to think of the benefits of MTL and try to get more understanding of the multi-task setting in UNMT's standard learning protocol.
  - [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/pdf/1711.02257.pdf), ICML 2018.
  - [Pseudo-task Augmentation: From Deep Multitask Learning to Intratask Sharing—and Back](https://arxiv.org/pdf/1803.04062.pdf), ICML 2018.
  - [An overview of multi-task learning](https://watermark.silverchair.com/nwx105.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAlAwggJMBgkqhkiG9w0BBwagggI9MIICOQIBADCCAjIGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMdyXlrfV1QXgS18hrAgEQgIICA1jYD3WLlUUgeDaNzuuCV2ohMMcK3y4k8XXu2AbFZ3gJz4hMeNAlq8UUpJYH4-wFsUg_3HeLzIg6XWdEdKqxgf9WsMYEsNz1NKYa-nA8wuNdsdKnNna73xrgSe0XsWMotb6rZetP9M2wIoa5evtyFhjYMVm0BW7JX0eM4WNHdNvZ4oI6JyhEpaKeefamiZk2S2DaD7crKI1b_qsdTN9Lei9FMIb_HCGW3i1DLGm2MAhW72i99KEO9aMSvM1uZkEMzL7swPYpPYwojKVAyax0tEzGcNWv7HrxbQvCE2ZavtVXXB5L1rgxwdc_TSuRXPWndwjl-XP24wtsM7m67IamSI9_vtaKXUQumcW6bVbvtXM6R15qfUvZsSgxUbXVvcqLiMgMZ4DldW0x4Ei2Sc-PHNPh-4iaFiVGvL0aanzHKQOGZSoPvDo6LXB-vk_W6caEcsx_AuXa8sJOWjIQp1eAxMb868XKvqH6Y2ijLe_-Vhjj_Lv5AQKtC10uGE1hBgZpfOEwZ_fU7f-Sg7_opgVSuo7N3MYIIFpsWBtEfXFBjVfLLMAJ0MjKXUQQIYHZ4tSXB3CVy-dbFi4_63UsYCaJhQGbkjLrhZJKyKuVk6WJZO_SNbKmCp3syQiyqnvR2ww792C0619APTclR6L68aSmeQEUQrME_PrBUpwtOS6RZGPmdu26), a review paper by Qiang Yang, 2017.



### Apr. 16

$$
\mathbb{E}_{c \sim \mathcal{C}} \mathbb{E}_{v \in \mathcal{V}} m(y^*, \tilde{y}_{arch, c, D}) \cdot \mathcal{1}[m(y^*, \hat{y}_{arch, c, D}) < m(y^*, \tilde{y}_{arch, c, D})]
$$

