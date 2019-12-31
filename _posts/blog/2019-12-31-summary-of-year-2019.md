---
layout: post
title: "A Summary of Year 2019: From research to life"
author: Guanlin Li
tag: diary
---

> When browsing the Twitter today, I find many people (especially AI researchers) are posting their new year summary and wish list (e.g. Yoav Goldberg were posting his [wish list](https://twitter.com/yoavgo/status/1205987145112051713) of 2020 for NLP research). Pointed out by my friend Changlong, I find [Chiyuan Zhang](http://freemind.pluskid.org/)'s blog very intriguing and motivating. As the first wish on my wish list, I wish to finish this blog post at the beginning moment of 2020, let's see...

#### Research: retrospective and perspective

2019 is the year for harvesting research outputs, I would like to use one word to summarize all, that is, ***luck***.

---

**My first first-author paper**

In Dec. 2018, I submitted a [paper](https://www.aclweb.org/anthology/N19-1046/) to NAACL after about one year's intern in Tencent AI Lab, Shenzhen, China, and it got accepted in Feb. of this year. The paper uses the hierarchical brown clustering tree to construct coarse-grained classification tasks for regularizing the intermediate representations of the 6-layer Transformer and obtain *very limited* BLEU improvement on larger corpus, i.e. WMT14 En$$\Rightarrow$$De.

I think the *only* shinning point of this paper is the understanding part, where we use the constructed psudo-tasks to probe the intermediate representations of the decoder and compare them with a random initialized network baseline and a linear baseline based on n-gram features. The <u>conclusion</u> is that higher layers have stronger performance on more complex psudo-tasks (the performance gap is evident) which implies the layer-wise enhanced representation towards the final fine-grained tasks (with $$\vert \mathcal{V} \vert$$ classes). This motivates the gradual task complexity regularization and prediction coherence regularization terms in the paper. 

In retrospective, actually, on the WMT14 dataset, our methods for training-from-scratch models does not work (compared to the baseline). So we use a pre-trained model and continue training with the regularization terms as well as the MLE loss on that model weights. This improvement is very tricky, since afterwards, we found that, continue training the same baseline model can get on-par performance around 29+ BLEU score on `newstest2014`, which indicates the uselessness of our methods. **However, as a reward for a year's struggling and chasing, it encourages me a lot.** 

Actually, this work could have done better if I really had analyzed the learned representation after our coherence regularization as [Lena Voita](https://lena-voita.github.io/) done from the perspective of [information bottleneck](https://lena-voita.github.io/posts.html) or [SV-CCA](https://arxiv.org/abs/1706.05806). Moreover, Brown clustering is one method for label clustering, but actually we can learn a hierarchical cluster through the training process and put such more fit-in gradual (emergent) structure prior into the training. Or more boldly, whether random partition of the label space can be benefit as constructing auxiliary tasks for confronting overfitting. Those are the unexplored part of the paper, and I wish one day I could find more formalisms or theories for me to restart doing them. *Theories should always come first, even if they are just logically reasonable hypotheses.*

**Challenge.** This paper also points out one challenge which all of us using autoregressive seq2seq paradigm should deal with, that is, the <u>train/test mismatch problem</u>. Since all the analyses are based on forced decoding over reference, it is hard to correlate the improvements gained through reference-based force-decoding to improvements over beam search (with exposure bias). I think a more elegant way or mechanism to make them actually correlated in theory would be very interesting and timely.

---

**My first analysis-directed paper**

After being luck at NAACL 2019, I was being luck again at EMNLP 2019, where my paper [Understanding DA](https://www.aclweb.org/anthology/D19-1570/) got accepted as short paper. Actually, this paper resulted from a failure experiments of a proposal which I dubbed as "Compositional Data Augmentation for NMT". The basic idea of that proposal is actually using the idea of [data as regularization](https://ieeexplore.ieee.org/document/726787) to incorporate the knowledge of compositionality into NMT. However, due to my premature experimenting skills, I failed to improve the performance effectively (with large BLEU gain) as other simple DA methods like Back-Translation. So at the beginning of March, Lemao discussed with me for an alternative choice, that is a survey like paper for DA which could clarify certain benefits or limitations of current state of DA methods. I accepted the proposal and then start to design fair experimental setting for all popular DA methods. After experimenting on augmenting the bilingual data, we started to design experiments to characterize the property of the augmented data $$\mathcal{A}$$ which distinguishing the original training set $$\mathcal{D}$$. We tried model robustness, data diversity, model exposure bias and find no regularities in the resulting statistics. And after realizing DA could potentially improve model generalization, we started to borrow something from the literature of understanding why DL generalizes and find some metrics reliable. Since the deadline were becoming closer, we did not have time to investigate a whole spectrum of generalization measures, thus resulted in a preliminary paper that focuses on input sensitivity and prediction margin. The experiments can largely support our arguments, but I still think for the sensitivity measure, we haven't done yet due to the discrete nature of our domain.

**Connections to data noising and knowledge distillation.** This work actually partially cleans the way towards analyzing DA for NMT. Currently, I think to separately analyze the methods in our paper is not a good entry point of understanding DA. Since RAML and SwichOut are actually adding random noise into data instead of relying on other learned knowledge (model-based DA), to divide DA methods into data noising (DN) and knowledge distillation (KD) could be more promising for systematic analyses. Intuitively, DN prevents overfitting while KD prevents fitting to noise in data, which are actually very different. Recent two [paper](https://jiataogu.me/publication/revisit-self-training)[s](https://jiataogu.me/publication/understand-distillation) from Jiatao Gu's group try to shed light on the above questions, which I think can be further improved.

**Challenge.** The previous challenge still exists since can those sensitivity or margin values links to model's inference behavior, who knows?

---

**Current research taste**

For my current research focus, I really want to do something with:

1. well-designed theory for understanding certain learning phenomenon or model behavior in a causal way;
2. or empirically well-motivated and designed measure that correlates well with the phenomenon and behavior;

Specifically, there is one biggest question hanging me for a very long time:

- Given a dataset $$\mathcal{D}$$ for training, and any model family $$\mathcal{M}_{\theta}$$, can we exactly determine the ***upper achievable generalization performance*** for the trained model given certain unseen test example $$\text{x}$$.

That is I want to decide in a model-agnostic way that is the dataset contain enough knowledge to generalization to certain $$\text{x}$$, which is a realistic fine-grained (possibly *statistical*) generalization test for the training data (training experience).

This can be very useful, since we can know whether we should add more data for training the model or just use smart online adaptation methods for improve the generalization performance. Like out-of-distribution detection, this can shed light on certified generalization given $$\mathcal{D}$$.

To shed light on that problem, I want to understand how a specific task structure (e.g. machine translation) hints on its algorithmic achievable generalization. How $$\mathcal{D}$$ enables generalization to what extent should be also investigated in a model-aware or (better) model-agnostic way.

> I think compositionality is a partial answer to the task of machine translation and other NLP tasks. And investigation under human-created approximation (like formal language) of natural language can be used to derive generalization guarantees.

#### Life: fantastic friends/hobbies and where to find them

During the time interning at Tencent AI Lab, I have encountered a lot of friends that really helped me to learn new things. 

[Xintong Li]() is one of my best friend here. He is like a technology geek, I have learned a lot about `linux`, `tmux`, `vim` and `latex`. *Simplicity* with *effectiveness* is his ultimate goal for creating stuffs or researching things. He is also a critical thinker, since everytime I chatted with him some research ideas or questions, he would help me develop a clarified and very grounding understanding of the problem at hand. This helps me a lot to connect abstractness to concreteness. His philosophy of simplicity also shape my current taste of research to find the simplest form of solution for understanding or improving. He is now a post-doc at Mike White's group at Ohio University. We will continue our friendship and collaboration.

[Yong Jiang]() is my friend and a figure to chase after, for his profound techniques and understanding in Machine Learning, which is very to my interests as well. He has a solid background in traditional NLP like parsing and other graphical model like modelling methods and dynamic programming. He is also my guide in life for tackling sentimental obstacles etc. He is now a researcher at Alibaba research, which I think is one of the best place for doing research with novel product exploration at the same time.

[Changlong Yu]() who is currently my friend who can chat with, walk with and discuss with. I learned how to live a work-life balanced life. Since he is an expert in knowledge-oriented NLP which I think is the most. challenging topics at this deep learning era, we chat a lot about how to use prior knowledge to regularise modeling learning, how to use symbolic knowledge to understand black-box model behavior and how to devise new knowledge-powered tasks for pushing forward the field of natural language understanding. Those chats are very engaging and I wish we can cooperate for a survey paper on this topic in the near future.

Other friends who I learned and will learn a lot from are Jiaxing Wang, Baosong Yang, Shilin He, Tianxiang Zhao, Ziyi Dou, Mengzhou Xia, Deng Cai, Huayang Li, Zhirui Zhang, Wenhu Chen et al. I wish them a brighter 2020.

#### Wish list

1. Get to tackle an important problem in NLP especially in NMT for generalization ability from a data-centered perspective.
2. Finish Judea Pearl's Book of Why and learn more about caulity for machine learning, which is a precious present from Xintong.
3. Get to work on the question "does *linguistic* structure benefits NMT and when" and finish a journal (possibly my most loved *Computational Linguistics*).
4. Get to work on an important machine learning problem (maybe at robustness and adversarial domain).















