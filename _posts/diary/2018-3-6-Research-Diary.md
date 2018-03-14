---
layout: post
title: "Research Diary from Mar. 6 - to Feb. 11"
author: Guanlin Li
tag: diary
---

### Mar. 6

"There are **two** ways to construct nonparametric models from PCFGs. **First**, we can let the number of non-terminals grow unboundedly, as in the Infinite PCFG, where the non-terminals of the grammar can be *indefinitely* refined versions of a base PCFG. Second, we can **fix** the set of non-terminals but permit the **number of rules or productions** to grow **unboundedly**, which leads to Adaptor Grammars."

"The new rule learnt by an Adaptor Grammar are compositions of old ones (that can themselves be compositions of other rules), so it is natural to think of these new rules as tree fragments, where each entire fragment is associated with its own probability."

Adaptor Grammar:

- Terminals V, non-terminals N
- Rules r and its probabilities p
- Concentration parameters $$\alpha$$, and each non-terminal $$A$$ has its own concentration parameter $$\alpha_A$$

"Adaptor grammars are so-called because they adapt both the subtrees and their probabilities to the corpus they are generating. **Formally, they are Hierarchical Dirichlet Processes that generate a distribution over distributions over trees that can be defined in terms of stick-breaking processes.** It is probably easiest to understand them in terms of their conditional or sampling distribution, which is the probability of generating a new tree $$T$$ given the trees $$(T_n \dots, T_n)$$ that the adaptor grammar has already generated."

#### Review of IBM models

Through [wikipedia](https://en.wikipedia.org/wiki/IBM_alignment_models). 

- **Model 1** is weak in conducting reordering or adding and dropping words. Two issues are associated with aligning: 1). reordering; 2). fertility (the notion that input words would produce a specific number of output words after translation). **(lexical translation)**
  - $$P(e, a\vert f) = \frac{\epsilon}{(\vert f \vert + 1)^{\vert e \vert}} \Pi_{j=1}^{\vert e \vert} P(e_j \vert f_{a_j})$$
- **Model 2** has an additional model for alignment that is not present in Model 1. The IBM model 2 addresses the shuffle invariance issue with absolute position alignment probability, $$P(i \vert j)$$, meaning  the probability that word at position j in source aligned to word at position i in target. Conceptually, this could functions as a prior of possible reordering of word at position j before seeing the real word itself. This probability is conditioned only on the target length $$l_e$$ and source length $$l_f$$. **(additional absolute alignment model)**
  - Two steps: lexical translation and alignment. 
  - $$P(e, a \vert f) = \Pi_{j=1}^{\vert e \vert} P(e_j \vert f_{a_j}) a(j \vert a_j, \vert e \vert, \vert f \vert)$$, is this formula a proper probability term? For given lengths $$\vert e \vert$$, $$\vert f \vert$$, sum over $$e_1, a[1]$$, where $$e_1$$ is all target vocabulary words and  $$a[1]$$ is are all $$\vert f \vert + 1$$ alignments. 
    - $$\sum_w \sum_i P(e_1, a[1], e_{2:}, a_{2:} \vert f) = \sum_w \sum_i \Pi_{j=2}^{\vert e \vert} P(j) P(e_1 \vert a_1) P(1 \vert a_1, \vert e \vert, \vert f \vert)$$
    - $$= \sum_i \sum_w P(w \vert f_{a_1}) P(1 \vert a_1, \dots) \Pi_2^{\vert e \vert} \dots$$
    - $$= 1 \cdot \Pi_2^{\vert e \vert} \dots$$
  - The representation of a is a list of tuple [(1, a_1), (2, a_2), ...], the list has length of the target sentence, and each tuple is a word alignment between target and source words. 
- **Model 3** **(extra fertility model)**




### Mar. 8

[12:57] Some papers on arXiv.

- [Analyzing Uncertainty in NMT](https://arxiv.org/pdf/1803.00047.pdf), Facebook AI Research team. 

  - Lack of understanding such as:

    - Large beam performance degradation
    - Under-estimation of rare words
    - Lack of diversity in the final translation

  - Inherent uncertainty of the task, due to the existence of multiple valid translations for a single source sentence, and to the extrinsic uncertainty caused by noisy training data. 

    - **Intrinsic uncertainty**
    - **Extrinsic uncertainty**

  - Their results show that search works remarkably well but that the models tend to spread too much probability mass over the hypothesis space.

  - **Model calibration** to alleviate the observation:

    - > **model distribution is too spread in hypothesis space.** 

- [Natural Language to Structured Query Generation via Meta-Learning](https://arxiv.org/pdf/1803.02400.pdf), NAACL short. 

  - Just use Finn's idea and design a new **relevance function** which is very suitable for the semantic parsing task, since query (SQL) is structured and highly typed ({Count, Min, Max, Sum ,Avg, Select}). 
  - Task is domain is grouped training examples. However the relevance function is heuristic thus fixed during training which might require improvements. 

- [Accelerating Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1803.02811.pdf), Peter Abbeel's group. 

  - A good summary for recent PG and Q-learning algorithms. 

[20:00] **Gamma distribution**

$$X \sim \Gamma(\alpha, \beta)$$, and set $$\lambda = \frac{1}{\beta}$$, then $$X \sim \Gamma(\alpha, \frac{1}{\lambda})$$. 

- Density form: $$f(x) = \frac{x^{(\alpha - 1)} \lambda^\alpha \exp^{(-\lambda x)}}{\Gamma(\alpha)}$$

- #### $x > 0$

- Additive property: if $$X_1 \sim \Gamma(\alpha_1, \lambda), X_2 \sim \Gamma(\alpha_2, \lambda)$$, then $$X_1 + X_2 \sim \Gamma(\alpha_1 + \alpha_2, \lambda)$$



### Mar. 10

[3:00pm] **BPE rethinking.** 

- <u>Joint BPE</u>: first merge the corpus, and then merge `vocab_size - one_gram_size` number. This will share two language with the common/intersected lexicon so as to make the shared lexicon have shared segmentation. 
- The BPE algorithm works as: after every greedy merge operation between the highest two frequency token A, B to AB, we should re-estimate the number of AB in the corpus and minus their counts into the count of A and B. 

### Mar. 11

[4:28pm] Some papers to read. 

- A gentle tutorial of the EM algorithm and its application to parameter estimation for GM and HMM. 


- Michael Collins' notes on IBM Models. 
- HMM based word alignment. 
- Nonparametric Word Segmentation for Machine Translation. COLING 10. 

