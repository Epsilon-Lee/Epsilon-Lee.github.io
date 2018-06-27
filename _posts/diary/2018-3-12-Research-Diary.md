---
layout: post
title: "Research Diary from Mar. 12 - to Feb. 19"
author: Guanlin Li
tag: diary
---

### Mar. 15

[16:34pm] What is word pieces?

- [Japanese and Korean Voice Search](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf), ICASSP 2012. Google Inc. 

The above paper gives a way to do Chinese, Korean or Japanese word segmentation to construct new word inventory by greedily increase the likelihood of the corpus. 

> "The wordpiece model is generated using as data-driven approach to maximize the language-model likelihood of the training data, given an evolving word definition."



### Mar. 16

[8:59am] My selected list of NAACL 20018 papers. 

- A structured syntax-semantics interface for english-AMR alignment. 
- Adversarial example generation with syntactically controlled paraphrase networks. 
- Are all languages equally hard to language-model?
- Behavior analysis of nli models: uncovering the influence of three factors on robustness. 
- Bootstrapping generators from noisy data. 
- Classical structured prediction losses for sequence-to-sequence learning.
- Delete, retrieve, generate: a simple approach to sentiment and style transfer. 
- Diverse few-shot text classification with multiple metrics. 
- Fast lexically constrained decoding with dynamic beam allocation for neural machine translation. 
- Neural machine translation decoding with terminology constraints. 
- Improving character-based decoding using target-side morphological information for neural machine translation. 
- Incremental decoding and training methods for simultaneous translation in neural machine translation. 
- Learning beyond datasets: knowledge graph augmented neural models for natural language processing. 
- Learning word embeddings for low-resource languages by PU learning. 
- Monte carlo syntax marginals for exploring and using dependency parses. 
- Natural language generation by hierarchical decoding with linguistic patterns. 
- Neural syntactic generative model with exact marginalization. 
- Supervised open information extraction. 
- When and why are pre-trained word embeddings useful for neural machine translation. 
- Neural particle smoothing for sampling from conditional sequence models. 

[4:16pm] Today and during weekends, my goal of research. 

- Most of the time, I should work on the coding of ibm-model based non-parametric alignment algorithm. 
  - IBM alignment model 1, 2 implementation. 
    - Understand EM through the implementation. 
    - Understand basic probabilistic assumptions made by IBM models. 
  - Code the forward-filtering backward-sampling algorithm. 
  - Think about the BPE initialization of the Chinese restaurant process of each source word. 
    - GAZA++ for word level alignment; then use BPE to segment words into subwords; and use this as the initial `alignments` and `e_segs` variable of our program. 

[4:33pm] Some papers to read through. 

- [Sparse Additive Generative Models of Text](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1214&context=machine_learning), ICML 2011. 
- [Stronger Baselines for Trustable Results in Neural Machine Translation](https://sites.google.com/site/acl17nmt/accepted-papers), first workshop for NMT 2017. 
- [Wild Approximate Inference](http://yingzhenli.net/home/pdf/wild_approx.pdf), Yingzhen Li. Talk slides. 
- [State Space LSTM Models with Particle MCMC Inference](https://openreview.net/forum?id=r1drp-WCZ), ICLR 2018, REJECT. 
- [Modeling relational data with graph convolutional networks](https://arxiv.org/abs/1703.06103), Ivan Titov's group. 
- [Syntax-Directed VAE for Structured Data](https://openreview.net/forum?id=SyqShMZRb), Le Song's group. 