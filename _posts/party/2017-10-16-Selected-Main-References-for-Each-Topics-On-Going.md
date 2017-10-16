---
layout: post
title: "Selected main references for each topics"
author: Guanlin Li
tag: party
---

[TOC]

> **Caveat.**
>
> This is an on-going list of essential paper based on coarse-grained topics. Note that, the following papers are not about a complete collection of all the related papers in the following main conferences (ACL, NIPS, ICML, ICLR, EMNLP, EACL, COLING), the criterion with which I select papers is to sieve less motivative and experimentally-weak papers and recommend those that are seminal, of high-quality, or at least well-written-for-understanding ones. 
>
> However, in terms of a general review of current trend, I list all the papers from 2017 conferences and mark the good ones with a start symbol (*). For papers before 2017, I just select according to the above  standard. 

### Neural Networks and Deep Learning

#### 1. Word2vec and Semantic Unit Representation

- (*)[Efficient Estimation of Word Representations in Vectors Space](https://arxiv.org/pdf/1301.3781.pdf), 2013. 
- (*)[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf), NIPS 2013. 
  - Original paper to efficiently learn word embeddings based on neural network language models. 
    - Continuous Bag-of-words or skip-gram with negative sampling or hierarchical softmax loss. 
- [Word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738), 2014. 
  - Tutorial paper, well-written to explain negative sampling objective and hierarchical softmax objective. 
- (*)[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), EMNLP 2014. 
  - Different from word2vec learning, which use SGD to learn local bias, but to learn global bias across corpus, so matrix decomposition method strikes back. 
- [Neural Word Embedding as Implicit Matrix Factorization](https://www.cs.bgu.ac.il/~yoavg/publications/nips2014pmi.pdf), NIPS 2014. 
  - Explaining learning mechanism behind Word2vec. 

#### 2. Attention Mechanism 

- (*)[Neural Machine Translationi by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), ICLR 2014. 
  - Original paper that propose attention mechanism for NMT. 
- (*)[Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166), EMNLP 2015. 
  - Solid work on trying different attention mechanisms. 
- [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130), ICLR 2017. 
  - Self-attention which has been proposed before, but this paper explicitly uses self-attention to do representation learning. 
- Some blogs for better understanding attention. 

#### 3. Memory Networks

- (*)[Memory Networks](https://arxiv.org/abs/1410.3916), 2014. ICLR 2015. Facebook AI Research. 
  - Original paper which motivates many memory-based deep learning architectures and paradigms, such as neural turing machine, one-shot learning etc. 
- (*)[End-to-End Memory Networks](https://arxiv.org/pdf/1503.08895.pdf), NIPS 2015. Facebook AI Research. 
- [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285), ICML 2016. Salesforce. 
- (*)[Key-Value Memory Networks for Directly Reading Documents](http://www.aclweb.org/anthology/D16-1147), EMNLP 2016. Facebook AI Research. 
- (*)[Neural Turing Machines](https://arxiv.org/abs/1410.5401), 2014. Google DeepMind. 
  - Original paper which motivates: 
    - many addressing mechanisms (kind of attention) for memory utilizing; 
    - many learning-to-program ideas in literature. 
- (*)[Hybrid Computing using a Neural Network with Dynamic External Memory](http://www.nature.com/nature/journal/v538/n7626/abs/nature20101.html?foxtrotcallback=true), Nature Oct. 2016. DeepMind. 
  - Nature paper on using Memory and Neural Turing Machine alike techniques. 
- [Pointer Networks](http://papers.nips.cc/paper/5866-pointer-networks), NIPS 2015. Google Research. 
- [Hierarchical Memory Networks](https://arxiv.org/pdf/1605.07427.pdf), rejected by NIPS 2016. MILA Bengio's group. 
  - Hashable memory. 
- [Neural Random-Access Machine](https://arxiv.org/pdf/1511.06392.pdf), ICLR 2016. Google. 
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279), ICLR 2016. DeepMind. 

#### 4. Gating Mechanism

[TO-DO]



### Topic models

[TO-DO]

#### 1. Latent Semantic Analysis 

#### 2. Latent Dirichlet Allocation

#### 3. Sampling-based Learning/Inference

#### 4. Variational Methods



### Sentiment Analysis

- (*)[Linguistic Regularized LSTM for Sentiment Classification](http://aclweb.org/anthology/P/P17/P17-1154.pdf), ACL 2017. 
- [Active Sentiment Domain Adaptation](http://aclweb.org/anthology/P/P17/P17-1156.pdf), ACL 2017. 
- [Best-Worst Scaling More Reliable than Rating Scales: A Case Study on Sentiment Intensity Annotation](http://aclweb.org/anthology/P/P17/P17-2074.pdf), ACL 2017. 
- [Contextual Bidirectional Long Short-Term Memory Recurrent Neural Network Language Models: A Generative Approach to Sentiment Analysis](http://aclweb.org/anthology/E/E17/E17-1096.pdf), EACL 2017. 
- [Attention Modeling for Targeted Sentiment](http://aclweb.org/anthology/E/E17/E17-2091.pdf), EACL 2017. 
- [Structural Attention Neural Networks for Improved Sentiment Analysis](http://aclweb.org/anthology/E/E17/E17-2093.pdf), EACL 2017. 
- [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](http://aclweb.org/anthology/D/D17/D17-1048.pdf), EMNLP 2017. 
- [A Cognition Based Attention Model for Sentiment Analysis](http://aclweb.org/anthology/D/D17/D17-1049.pdf), EMNLP 2017. 
- [Towards a Universal Sentiment Classifier in Multiple Languages](http://aclweb.org/anthology/D/D17/D17-1054.pdf), EMNLP 2017. 
- (*)[Tensor Fusion Network for Multimodal Sentiment Analysis](http://aclweb.org/anthology/D/D17/D17-1116.pdf), EMNLP 2017. 
- (*)[Document-level Multi-Aspect Sentiment Classification as Machine Comprehension](http://aclweb.org/anthology/D/D17/D17-1216.pdf), EMNLP 2017. 
- [Capturing User and Product Information for Document Level Sentiment Analysis with Deep Memory Network](http://aclweb.org/anthology/D/D17/D17-1055.pdf), EMNLP 2017. 
- [Sentiment Lexicon Construction with Representation Learning Based on Hierarchical Sentiment Supervision](http://aclweb.org/anthology/D/D17/D17-1053.pdf), EMNLP 2017. 
- [Refining Word Embeddings for Sentiment Analysis](http://aclweb.org/anthology/D/D17/D17-1057.pdf), EMNLP 2017. 



### Knowledge Graph, Relation Extraction

[TO-DO]



### Text Generation, Neural Machine Translation

[TO-DO]



### Representation Learning and Domain Adaptation

[TO-DO]



### QA-based Dialogue/Dialogue-based QA

- [Key-Value Retrieval Networks for Task-Oriented Dialogue](http://www.aclweb.org/anthology/W17-5506), SIGDIAL 2017. 
- (*)[Reading Wikipedia to Answer Open-Domain Questions](http://cs.stanford.edu/people/danqi/papers/acl2017.pdf), ACL 2017. 
- (*)[Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access](https://www.aclweb.org/anthology/P/P17/P17-1045.pdf), ACL 2017. 
- (*)[Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://www.aclweb.org/anthology/P/P17/P17-1061.pdf), ACL 2017. 
- (*)[Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses](https://www.aclweb.org/anthology/P/P17/P17-1103.pdf), ACL 2017. 
  - Evaluation of dialogue system is notoriously hard. How about learning an evaluation metric? However the same question remains, what information bias can the model learn? 
- [Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://www.aclweb.org/anthology/P/P17/P17-1163.pdf), ACL 2017. Cambridge Univ. 
- [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://www.aclweb.org/anthology/P/P17/P17-1046.pdf), ACL 2017. MSRA. 
- [Adversarial Learning for Neural Dialogue Generation](https://www.aclweb.org/anthology/D/D17/D17-1229.pdf), EMNLP 2017. 
- [Towards Implicit Content-Introducing for Generative Short-Text Conversation Systems](https://www.aclweb.org/anthology/D/D17/D17-1232.pdf), EMNLP 2017. 
  - Hierarchical gated fusion unit. (Architecture design)
- [Affordable On-line Dialogue Policy Learning](https://www.aclweb.org/anthology/D/D17/D17-1233.pdf), EMNLP 2017. 
  - Human-in-the-loop. 
- [Generating High-Quality and Informative Conversation Responses with Sequence-to-Sequence Models](https://www.aclweb.org/anthology/D/D17/D17-1234.pdf), EMNLP 2017. 
  - Self-attention, single turn. 
- [Composite Task-completion Dialogue Policy Learning via Hierarchical Deep Reinforcement Learning](https://www.aclweb.org/anthology/D/D17/D17-1236.pdf), MSR. 
  - Hierarchical RL, task-oriented dialogue. 
- [Agent-aware Dropout DQN for Safe and Efficient On-line Dialogue Policy Learning](https://www.aclweb.org/anthology/D/D17/D17-1259.pdf), EMNLP 2017. 
  - Human-in-the-loop. 
- [Bootstrapping incremental dialogue systems from minimal data: the generalization power of dialogue grammar](https://www.aclweb.org/anthology/D/D17/D17-1235.pdf), EMNLP 2017. 
  - Experimental paper, good point to use compositionality of grammar to better generalize. 


- [Teaching machines to converse](https://github.com/jiweil/Jiwei-Thesis), Jiwei Li's **PhD thesis** from Stanford Univ. 10/13/2017. 

