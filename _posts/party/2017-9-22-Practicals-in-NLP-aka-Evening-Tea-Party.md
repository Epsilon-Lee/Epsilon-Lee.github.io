---
layout: post
title: "Practicals in Natural Language Processing aka. Evening Tea Party"
author: Guanlin Li
tag: party
---

>  Drafted by Guanlin Li. 

### 1. Motivation and Goal

Natural Language Processing (NLP) is a more practical subject lies on the boundary of science and technology. Many beginners of NLP, including myself, get really confused of the models and algorithms when dealing with similar situations in practice on their own. The confusion will lead to longer time of exploration, and slow down learning or even end up with people's surrender or give-up. It is a sad story. 

The above mentioned sad stories are mainly caused by the gap between theory and implementation, which is this **Evening Tea Party**'s main motivation and meaning about. More specifically, we are: 

- **Aiming at discuss and learn about basic techs in NLP with implementation details**

Another main specialty of our Evening Tea Party is that the party hope to be a avenue for chatting, discussion, creativity, imagination and of course happiness and joy. So we would like to find a better place to hold our party and sponsor. 

> **Note.** Currently, we would like to settle at <del> 'Guangxi Cafe Room' at second floor of Zonghe Building</del>618 Room at the sixth floor of Xin Ji Shu Building. If you have a better place to recommend, feel free to contact Guanlin Li.

### 2. Topics and Materials

The tentative topics and materials are all from Graham Neubig's course [NLP Programming Tutorial](https://github.com/neubig/nlptutorial). He is an awesome researcher and programmer for his contribution in machine translation and a very fast deep learning framework [DyNet](http://dynet.io/). The topics are compacted as following. 

1. N-gram Language Models (LMs)
   - Math concepts of statistical/probabilistic model, likelihood, maximum likelihood estimation (MLE), entropy and perplexity. 
   - Statistical LMs and its estimation (Uni-gram/Bi-gram models). 
   - Smoothing techniques. 
2. Word Segmentation
   - Viterbi algorithm and forward-backward stages. 
   - Word segmentation based on an estimated bi-gram language model. 
3. Part-of-speech Tagging with Hidden Markov Models
   - The concept of generative models, and Hidden Markov models (HMM). 
   - MLE for estimating parameters of HMM. 
   - The concept of decoding in probabilistic models (probabilistic graphical models). 
   - Using Viterbi algorithm to perform decoding in HMM. 
4. The Perceptron Algorithm and other Discriminative Models
   - Perceptron and its on-line learning algorithm. 
   - Stochastic gradient descent and logistic regression. 
   - The concept of margin and support vector machine. 
   - Margin based perceptron and L1, L2 regularization based perceptron. 
5. Neural Networks and Deep Learning for NLP
   - TBD [Tentative: PyTorch/Tensorflow basics, and a sentiment classification use case.]
6. Topic Models
   - Review the concept of supervised and unsupervised learning. 
   - Topic modeling as an unsupervised learning, Latent Dirichlet Allocation (LDA). 
   - MLE for LDA. 
   - Gibbs sampling and the learning of LDA. 
7. Parsing Techniques
   - Phrase structure tree parsing. 
     - Phrase structure grammar (Context-free grammar). 
     - Probabilistic modeling of phrase structure grammar and probabilistic context-free grammar. 
     - The concept of hypergraph. 
     - Viterbi algorithm for finding highest scored graph and inside-outside algorithm for finding highest scored hypergraph. 
     - CKY algorithm. 
   - Dependency tree parsing. 
     - Shift-reduce based method. 
     - (*) Spanning tree based method. 
8. Structured Perceptron and Search Algorithms
   - Structured perceptron training for HMM. 
   - The concept of intractable search problems; Beam search for intractable decoding. 

The above topics are tentative and we should not follow them strictly. However our ultimate goal is to understand the implementation part of those algorithms, so you should remember one thing: 

> **A detailed understanding of the algorithm is enough to understand implementation, but not enough to analysis experiment result.** 

So it is enough to not totally understand the underlying model, since sometimes in practice, you only need to run some demo program or toolkits for doing the algorithm for yourself. However, always remember the caveat that **understanding is the most beautiful thing in the world that you should struggle to achieve**! 

### 3. Party Programme

**2** or **3** members are required to *collaborate* to prepare **each** party meeting. We would like to meet **once a week**, maybe every **Friday** night. 

#### 3.1 Preparation Time

Preparation **can** take **2** weeks, **one** week for conceptual understanding and basic coding; and **another** for real implementation and experiment analysis. Preparation is the most valuable time for speakers, it basically flows like this. 

![]({{ site.baseurl }}/public/img/party/prepare_workflow.png)

The hopeful effects of preparation: 

- **Understanding** the model and its **implementation**, and **run** the code on some given **datasets**. (*Note that, Neubig gives some test data in directory named `data` and `test` for training and testing repectively, that's awesome, hah!*)
- **Discussion** makes deeper understanding of your problem, but at first, try to solve your problem independently through Google. Then, discuss more!
- **Implementation** really takes pain in the beginning, however, it pays back at the middle of our learning process, *I promise*. So don't be shy to code the algorithm. Use **python** first, and search for grammars and other [syntax sugar](https://zh.wikipedia.org/zh-hans/%E8%AF%AD%E6%B3%95%E7%B3%96) for python through **Google**! 
- Coding up is **just the beginning**. See the experiment results, try to visualize results or intermediate outputs of the algorithm will make you start to think about the property in your data and the nature of your code. Your future work (if engineering oriented) is **all about** analysis of data and results from experiments which is data as well! 
- Try to **build up your voice** before delivering your speech. Communication/speech ability is important in your job career. 

#### 3.2 Presentation Time

Presentation should flow like this: 

![]({{ site.baseurl }}/public/img/party/presentation_workflow.png)

The hopeful effects of presentation: 

- For the speaker. 
  - Don't be shy, because you are so cool for that you are our super star of that night! And speaks loudly to make sure every hears you. 
  - Try to make some appropriate derivations on white board, if someone ask you about the math. 
  - Don't just eat pizza while speaking. 
- For the listeners. 
  - Don't be shy, because you are so cool for that you are supporting your favorite super star that night! **So ask questions please.** 
  - Don't just eat pizza while the speakers are eating! 

### Appendix

#### A. Group Member

Currently we have a moderate party size, it is a mixture of many lovely people at Harbin Institute of Technology, mostly are Master students. Some of them are from Labs other than *Machine Intelligence and Translation*, like [*Social Computing and Information Retrieval*](http://ir.hit.edu.cn/). following are the total list of people: 

> 白雪峰 鲍航波 陈双 龚恒 候宇泰 胡东瑶 李冠林 马晶义 田恕存 王贺伟 王瑛瑶 杨奭喆 吴焕钦 张冠华 张丽媛 赵晓妮 赵笑天 

`Caveat: Tell me if I miss anyone.`

#### B. Schedule (Updated on 9/23/2017)

| Topics                                  | Members                | Date  |
| --------------------------------------- | ---------------------- | ----- |
| 1. N-gram Language Models               | 田恕存，龚恒，鲍航波             | 09/29 |
| 2. Word Segmentation                    | 白雪峰，杨奭喆，邓俊锋            | 10/13 |
| 3. POS Tagging with HMM                 | 胡东瑶，赵笑天，邓俊锋            | 10/20 |
| 4. Perceptron and Discriminative Models | 田恕存，龚恒，鲍航波             | 10/27 |
| 5. Neural Networks and Deep Learning    | 白雪峰，王瑛瑶，吴焕钦，王贺伟，李冠林，陈双 | 11/03 |
| 6. Topic Models                         | 马晶义，张冠华，赵笑天            | 11/10 |
| 7. Parsing Techniques                   | 侯宇泰，陈双，胡东瑶，张冠华，李冠林，马晶义 | 11/17 |
| 8. Structured Perceptron and Search     | 胡东瑶，李冠林，赵晓妮            | 11/24 |

