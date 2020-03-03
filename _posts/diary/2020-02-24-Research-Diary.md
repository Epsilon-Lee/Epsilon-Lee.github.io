---
layout: post
title: "Research Diary from 2/24/2020 - 2/28/2020"
author: Guanlin Li
tag: diary
---

- toc
{:toc}


### Feb. 24 2020

> It's Monday today; nice sunny turns into cloudy. ⛅️ 

#### Plan the day

1. Modify the code so that it could be run with MASS checkpoint;
2. Visualize the co-training instances constructed by BT along the first 1000 iterations;
   - Try to give intuition to the Sec. 2.4's *back-translation hypothesis*;
3. Read the Co-Training theory papers (Blum's and Balcan's papers) and take notes;

#### Miscs

- [Gregory Gundersen's blog on log-sum-exp trick](http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/).



### Feb. 27 2020

> It's Thursday; cloudy to sunny ^_^

1. (EXP) Fix checkpoint loading for MASS, including loading `dec_enc_attn`, which is currently missing; [`null`]
2. (EXP) Start training from scratch the `joint emb5M` initialization method; [`x`]
3. (EXP) Run based on `pretrain.emb.5M`, `pretrain.model.5M`, `pretrain.model.fb` and `pretrain.model.mass`, add hook during BT process, and gather the BT constructed training data. [`x`]
4. (IDEA) Get a ***specific*** <u>research agenda</u> done with **PLAN** till next Friday; [`x`]

#### Research Agenda on `DLUNMT`

> This part is organized with specific research questions (RQ) and its corresponding hypothesis (RH) or answers (RA).

`[TO-DO]`



### Feb. 29 2020

> Friday, cloudy currently. (8:38am)

1. Finish the above not finished part;
2. Process NIST into BPE and run the code, using `nist02` as validation set;





















