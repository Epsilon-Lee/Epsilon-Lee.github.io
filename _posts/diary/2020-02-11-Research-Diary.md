---
layout: post
title: "Research Diary from 2/11/2020 - 2/21/2020"
author: Guanlin Li
tag: diary
---

**Table of Contet**

* [11-2-2020](#11-2-2020)
* [15-2-2020](#15-2-2020)
* [19-2-2020](#16-2-2020)

### 11-2-2020

1. Summarize the core of my last week work;
2. Prepare for talking to *lemao*;

---

#### Summary of last week

I divide the progress made last week into two parts: i) experimental progress and ii) conceptual progress. Hopefully it will make me plan and have a better gain this week.

**Experimental progress**

1. How to use and configure `jizhi` platform for my experimental environment;

   - Installing `tmux` and `vim` takes about 4 hours, and on `jizhi`, there is ***no need*** to install `tmux`,  so only take time to configure `vim`;

   - `apex` does not work for me currently (which can make experiment faster due to mixed precision computation on `f16`);

   - `fastText` python package is used for loading pretrained embedding, but the python bind works weirdly: the `brew unlink gcc` trick;

   - Multi-GPU training is crucial for performance and speed; currently not using `apex` but `torch.nn.DistributedDataParallel` with `accumulate_gradients` ***not*** working (so larger batch with limited GPU numbers is not possible);

   - The multi-GPU running script trick:

     ```bash
     export NGPU=4; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py ...
     ```

2. Run default experiment on `en-fr` dataset with size of each language ***5M***:

   **a.** pretrain the model on 5M, and fine-tune on 5M;

   **b.** use the FB provided pretrained model on 26M (?), and fine-tune on 5M;

   **c.** use pretrained embedding via `fastText` on 5M, and fine-tune on 5M;

   **d.** use the MS provided pretrained model on (data?), and fine-tune on 5M;

   > Experimental results will be posted on `overleaf` paper appendix's experimental section. And currently, the results are as expected, BLEU performance **b. > a.  > c.**.

3. Become basically familiar with the `XLM` codebase;

   - Familiar with:
     - Basic functional flow of the codebase;
     - How to load pretrained model, embedding; how to load from checkpoint;
   - Not familiar with:
     - Logging mechanism (***to add logging for early stage loss visualisation***);
     - Data iterator mechanism (***to build iterator for 10 batches valid bilingual data***);
     - Model selection mechanism;

**Conceptual progress**

![](../../public/img/fig/dlunmt-snapshot/Feb-2.jpeg)

### 15-2-2020

1. Read reviews carefully again, summarize key issues and questions; then write two versions of response to R1;
2. Write response to R2 and R3 and show respect;



### 19-2-2020

> Back from rebuttal to my own work.

If I ask for *1-week*'s **<u>free time</u>** for investigating papers and preparing solutions with respect to the above blueprint, what would I actually investigate?

**Part A - On initializations**

- <u>Emerging Cross-lingual Structure in Pretrained Language Models</u>, Shijie Wu et al. JHU & Facebook AI.
  - First they should prove them effective on cross-lingual transfer; HOW?
  - Hypothesize factors and study their influence: shared params in the top layer of the multilingual encoder;
    - No need to share vocabulary; (Easy to implement?)
    - Monolingual training can come from very different domain; HOW?
- <u>The Missing Ingredient in Zero-Shot Neural Machine Translation</u>, Google AI.
  - The representational invariance across languages part should be read thoroughly;
  - Sec. 4 "aligning latent representations", the solution that they proposed, similar to universal language or interlingua?
  - In Sec. 5.2 "Quantifying the Improvement to Language Invariance"; HOW?
- <u>Evaluating the Cross-Lingual Effectiveness of Massively Multilingual Neural Machine Translation</u>, Google Research.
  - Same author as the above one.
  - They study cross-lingual tranferibility of multilingual NMT model on downstream tasks.

- <u>Multilingual Denoising Pre-training for Neural Machine Translation</u>, arXiv Jan. 22 2020.
  - To see how this paper make the argument through empirical experiments:
    - "*We also show that language not in pretraining corpora can benefit from mBART, strongly suggesting that the initialization is at least partially language universal*"
- <u>Multilingual Alignment of Contextual Word Representation</u>, ICLR 2020 from Dan Klein's group.
  - "*These results support contextual alignment as a useful concept for understanding large multilingual pre-trained models*"
  - "*The degree of alignment is causally predictive of downstream cross-lingual transfer, contextual alignment proves to be a useful concept for understanding and improving multilingual pretrained models.*"
  - "*Contextual word retrieval also provides useful insights into the pre-training procedure, opening up new avenues for analysis*"
- ***Some other papers:***
  - <u>On the Difficulty of Warm-Starting Neural Network Training, Ryan Adams' group at Princeton</u>.
    - Learn how they carry out experiments, the dimensions of analyses they conducted.
  - <u>Why Does Unsupervised Pre-training Help Deep Learning?</u>, JMLR 2010.



**Part B - On co-training (denoising)**

- <u>Combining Labeled and Unlabeled Data with Co-Training</u>, COLT 1998.

  - "*learning with lopsided misclassification noise*"
- <u>Co-Training and Expansion: Towards Bridging Theory and Practice</u>, NeurIPS 2005.
  - Sec. 4 "*Heuristic Analysis of Error Propagation and Experiments*".
  - $$(<x_1, x_2>, l)$$, and existence of two concept classes $$c_1(x_1) = l = c_2(x_2)$$: each example contains two views, and each view contains sufficient information to determine the label of the example.
  - *Iterative Co-Training*, $$h_1: \mathcal{X}_1 \mapsto \mathcal{D}^l, h_2: \mathcal{X}_2 \mapsto \mathcal{D}^l$$
  - Co-Training effectively requires two distinct properties of the underlying data distribution in order to work:
    1. There should at least in principle exist low error classifiers $$c_1, c_2$$ on each view;
    2. These two views should not be too highly correlated, i.e. we need to have at least some examples where $$h_1$$ is confident but $$h_2$$ is not for the co-training algorithm to actually do anything.
- <u>Improving Generalization by Controlling Label-Noise Information in Neural Network Weights</u>, ICML 2020 submitted.
  - 
- ***Some other papers*** that may shed light on ***"noise-signal balance"*** (Noise-Signal Rate)
- <u>Understanding why neural networks generalize well through GSNR of parameters</u>, ICLR 2020.
  - Gradient signal-to-noise ratio (GSNR): $$\frac{\text{Gradient's squared mean}}{\text{Gradient's variance}}$$ over the data distribution (at certain model update timestamp $$t$$?)
  - They also show SGD tends to lead to larger GSNR value than shallow models.
    
  - <u>Think Global, Act Local: Relating DNN generalisation and node-level SNR</u>, arXiv Feb. 11 2020.
  - SNR calculated for a trained DNN to as surrogate for measuring its generalization performance.









### 21-2-2020

> It's Friday, the EOW.

#### Xintong's recommended workflow using overleaf

1. Create a project, every collaborator `git clone` that project; then divide their work to not focus on the same part of the document;
2. During proof reading, everyone meets online (with possible video connection), and edit on overleaf;
3. Finally, one validate all details and submit the paper.

And other tips from him:

>  za (toggle模式), zm (fold), zr (unfold), zf (any), zd
>
> :h fold - to see all the instructions







