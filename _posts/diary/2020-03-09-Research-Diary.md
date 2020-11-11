---
layout: post
title: "Research Diary from 3/09/2020 - 3/13/2020"
author: Guanlin Li
tag: diary
---



- toc
{:toc}




### Mar. 9 2020

> It's Monday but cloudy.

#### 上周小结

- Fairseq:
  - 多reference训练卡死代码调试完成；（已不重要，237服务器code已找回）
  - LCA训练过程卡死代码调试完成；
  - 整理出一份给两位师弟跑LCA的（基于润泽和朱老师的code）轻量级LCA code，这周把代码阅读和最基本的实验任务分配下去；
- Dymestifying Learning of UNMT (DLUNMT)
  - 复现emnlp 18论文中ablation实验：与论文中结果一致，即在后续根据DAE (Denosing Auto-Encoding)+BT进行调优过程中一定要加入BT损失，并且证明即使使用XLM的初始化，不加入DAE损失也会使得模型完全训练失败（BLEU<1）；但MASS的初始化方法去掉DAE之后并不会出现此问题；具体讨论见下面实验设计节：[DLUNMT Research Agenda](#dlunmt-research-agenda).
  - 我们工作的实验设计 (目前仅有一些research questions，对应的假设，与验证实验设计，还没有成型论文的骨架，可根据假设证实与证伪一步步推进)
- Generalization Barriers
  - 重新熟悉并跑起来risk estimation的算法；
- Misc:
  - 审稿、服务器配置；

#### DLUNMT Research Agenda

---

---

##### 主干研究问题

- 为什么UNMT的训练能够成功？

  > 反面问题：为什么会存在完全失败的训练设定？

##### 背景

- <u>术语定义</u>
  
  - **独立预训练**：seq2seq模型的编码-解码器网络*各自独立构造*自监督（Self-supervised）损失进行训练，独立得到各自的参数；具体来说有下面几种模式：
    - 均独立随机初始化；
    - EMNLP best paper等embedding初始化策略: 联合得到shared vocabulary情况下，embedding部分的参数初始值，并分别用于初始化编码器的输入embedding矩阵，解码器的输入与输出embedding矩阵；
    - XLM类初始化策略：基于自监督损失直接训练类语言模型的神经网络，并由编码-解码器的对偶结构，使用同一预训练参数分别初始化编码与解码器（解码器到编码器表示的注意力机制相关参数只能随机初始化）；
  - **联合预训练**：seq2seq模型的编码-解码器网络联合构造自监督损失进行训练（解码器到编码器表示的注意力机制相关参数同样得到预训练）；已发表的此类方法诸如：
    - seq2seq补全（infilling）预训练：MASS [3]
    - seq2seq去噪（denoising）预训练：mBART [1]
- <u>我的实验证据</u>
  
  - 基于不同的初始化或预训练方法（`random`, `jointEmb`, `XLM`, `MASS`），**仅使用BT损失**进行训练:
    - `random`, `jointEmb`, 以及`XLM`中 FB的SOTA checkpoints，均训练失败，BLEU<1;
    
    - `MASS`训练成功，BLEU为SOTA；
    
      > **重要推论**
      >
      > 上面两个现象说明：
      >
      > - 仅仅依靠之前我们提到的Co-Training（即on-the-fly BT loss）的思路
      >   - 即：两个对偶编码-解码器模型在有效监督信号的——例如通过归因得到的有效词对齐的——覆盖率Coverage、差异性Diversity
      >     - 覆盖率越大，差异性越大训练越能顺利进行；
      > - 去解释训练是否成功，而不考虑DAE损失的作用，是不可取的，因为实验证明，去掉DAE后，训练会完全失败. 但有DAE后，DAE+BT的训练方法，即使在整个编码-解码器模型都随机初始化的情况下，仍然能训练得到初步的翻译性能（BLEU：～13）
- <u>他人的实验证据</u>
  
  - [2]中ablation，在`jointEmb`情形下，去掉任何一个损失（DAE或BT），调优训练均失败（BLEU~0）;
  - [3]中采用DAE进行**预训练**，并继而用DAE+BT进行调优得到的结果比XLM低3-4个BLEU点，但训练是成功的，[3]并没有汇报仅使用BT调优的结果（估计会失败）；
  - 最新FB arXiv论文[1]中在多语的设置下进行DAE联合预训练，并继而仅仅使用BT进行调优，翻译性能与MASS相当；值得注意，在调优初始阶段，采用受限的BT进行on-the-fly的解码，限制为token仅来源于目标语言；

---

---

##### 研究问题一

具体涉及下列两个问题：

1. 为什么DAE如此重要，丢掉DAE损失，任何（独立）初始化的方法在仅BT损失训练下，BLEU值也会趋于0（训练失败）？
2. 为什么DAE对于**联合预训练**（主要指MASS）的模型初始化方法可以舍弃？

**假设**（Hypothesis）

- **假设一**：DAE（在调优阶段作用于整个编码-解码器模型）起着**类似**联合预训练的作用
  - **验证实验设计**
    - 在给定独立预训练得到的初始化模型后，<u>用DAE损失进行联合预训练</u>，并在此过程中监测翻译性能（BLEU）是否有提升，若有以此为预训练终止的标准，若没有根据validation loss终止预训练；
    - 基于上述DAE进行联合预训练的模型，仅使用BT损失进行调优，观测训练是否仍旧会失败；
- **假设二**：DAE可以替换成其他联合预训练策略（即<u>**联合训练**</u>是仅使用BT损失便可获得调优成功的关键）
  - **验证实验设计**
    - 证明在**多任务**（即DAE与BT损失**共存**的经典训练协议）设置下，DAE可被MASS替换；

**结论**

- DAE在经典UNMT训练协议中起到的作用和**联合预训练**的作用**一致**或**不一致**；

- DAE在预训练阶段的作用与MASS相似或MASS的作用**包含**DAE的作用（DAE$\subset$MASS） $\Rightarrow$ 表象上，能使得编码-解码器模型具有**初步的**双语翻译能力（即有一定BLEU值，例如10左右，这便也初步奠定了Co-Training能成功的基础，两个**弱**对偶翻译模型）；

  > 基于[2]中附录的ablation实验，可见去掉BT损失后的训练协议也会使得训练失败，这与MASS的github repo中所给的EN-DE预训练日志最后得到的具有基本翻译能力的模型还是有极大区别的，可推测DAE与MASS不尽相同. 但由[1]知，DAE作为**多语**联合预训练的方法，应用于仅采用BT损失作为调优的UNMT训练是成功的，注意到[1]中描述仅仅使用BT训练进行训练数据生成时的一个技巧，“However, we do constrain mBART to only generating tokens in target language for the first 1000 steps of on-the-fly BT, avoid it simply copying the source text.”

---

---

##### 研究问题二

若**“研究问题一”**结论为***一致***，那么进一步地：

1. DAE既然与联合预训练作用相似或一致，那么，DAE能否取代独立预训练呢？如果能，那么多大程度上能取代独立训练呢？[1]提供了一方面的证据.

2. 在多任务设置下，DAE（或MASS）是如何避免仅使用BT损失的训练协议陷于失败的呢？

   > 该问题可归约为回答联合预训练的本质优势的问题：
   >
   > - 原因一：是因为能使得模型有基本的双语翻译能力么？
   > - 原因二：是因为能防止某些退化或奇异的情形发生么？

**假设**（Hypothesis）

- **假设一**：DAE可以取代独立训练，但在联合训练后的BT调优的开始阶段，**须**限制on-the-fly生成与源端**相异**语言的句子，从而限制**退化为copy**的现象
  - **验证实验设计**
    - 采用DAE进行联合预训练；
    - 在调优阶段，仅使用BT损失进行训练，且在下面两种设置下进行：1）on-the-fly阶段不限制输出token来自的语言；2）on-the-fly阶段限制输出token来自的语言为目标语言；
- **假设二**：BT的训练失败是由于在调优之初，on-the-fly生成的Co-Training训练样例的某种**奇异现象**（例如退化为copy源端），使得在**不加其他约束损失**的情况下，基于Co-Training训练样例的训练，让模型陷入某种学不到任何**有效监督信号**的情况中
  - **验证实验设计**
    - 对比第一次迭代训练过程中（至第一次evaluation为止）产生的on-the-fly样例，看是否出现退化现象；

<del>独立预训练有效，但，独立预训练的作用是什么呢？</del>

---

若**“研究问题一”**结论为***不一致***，则应该转向思考：

- DAE+BT的多任务训练设定的**必要性（对于独立预训练为必要条件）** $\Rightarrow$ 双语信号是如何在多任务训练中出现的？

**假设**（Hypothesis）

- 事前解释（prescribed explanation or theory）

  - `TO-DO`

- 事后解释（post-hoc explanation）：DAE一方面防止BT陷入奇异的模型参数配置，另一方面辅助BT产生***有效监督信号***（通过有效的词对齐可量化）；并在Co-Training过程中由于对偶二模型的**有效**翻译能力的**差异**可互相**互补地**<u>扩充</u>有效的监督信号直到DAE+BT的作用饱和.

  - **验证实验设计**

    - **定义**（词对齐全集）
      - **类别一（model-agnostic）**给定双语语料$\mathcal{B}$，通过GIZA++的词对齐模型，为双语数据$(x, y) \in \mathcal{B}$标注次对齐$a$，得到有次对齐标注的双语语料$\mathcal{B}^a$；
      - **类别二（model-related）**给定相同初始化后$k$次无监督训练得到的模型$\mathcal{\hat{M}}_k$，在双语语料$\mathcal{B}$上根据attribution策略，得到各样例的词对齐标注，进而得到$\mathcal{B}^a$；
    - **定义**（有效监督信号）给定$t$时刻模型$\mathcal{M}^t$，与**隐含**数据$x, y$（实际训练中只可见$x$），翻译的结果为$\hat{y}=\mathcal{M}(x)$，以及$t'$时刻模型$\mathcal{M}^{t'}$（$t'>t$），有效监督信号定义为满足下列二条件的$x, \hat{y}$中的词对$x_i, \hat{y}_j$：
      - $A[\hat{y}_j]_{\mathcal{M}^{t'}} = x_i$，且$x_i, \hat{y}_j \in a^{x,y}$，这里$A[\cdot]_{\mathcal{M}^{t'}}$为模型在$t'$时刻的**归因算法**得到的源端与目标端$\cdot$对齐的词，$a^{x,y}$为词对齐全集定义中得到的每个隐含数据的词对齐标注；
      - ***高效计算***：在训练过程中，保留t时刻的batch B（BT产生的数据），在t'时刻，在上述保留的batch B上计算有效监督信号，计算完毕后可丢弃batch B，$\delta(t) = t' - t$为超参数，另一层面，也可以理解为模型的学习速度，即$\delta(t)$越短，模型学习越快；
    - **定义**（噪声监督信号）依照上述有效监督信号的定义，我们可以延伸出噪声监督信号的定义，即被$t'$时刻模型学到的$t$时刻模型产生的非对齐数据；
    - **定义**（信噪比）若t'时刻的有效监督信号为个数为$\vert a_{e} \vert$，噪声监督信号个数为$\vert a_{n} \vert$，我们有$\vert a \vert = \vert a_e \vert + \vert a_n \vert$，信噪比定义为$SNr = \frac{\vert a_e \vert}{\vert a_n \vert}$

    实验中对上述信噪比进行计算，DAE+BT的训练可有效的提升$SNr$，但仅仅BT的训练却只能维持很低的$SNr$，使得训练无法朝着期望的方向前进 

---

---

##### *研究问题三

1. 任何联合预训练的方法都能使得该对偶的模型参数化方式在自监督训练过程中获得初步的双语翻译能力么？
2. 实验证明MASS能获得这种初步的双语翻译能力，背后的原因是什么呢？

---

---

##### Reference

*** [1]. Multilingual Denoising Pre-training for Neural Machine Translation, arXiv Jan. 22 2020, FAIR.***

> 对于[1]提出的多语去噪预训练（mBART），可以理解为UNMT的经典训练协议（learning protocol）中的DAE部分，所以[1]给我们的启发是：UNMT的经典训练协议中DAE其实一定程度上起着联合与训练的作用.
>
> **注**：mBART中的DAE与unmt经典训练协议中的DAE损失构造的构造不完全一致，差异主要源于噪声函数$g(x)$的定义：
>
> - [1]中主要为两种：nh
>   - 给x中一个span挖去$x_{\setminus i-j}$，然后编码器编码之，并自回归的预测完整$x$；
>   - 对$x$的字部分单词打乱顺序，然后类似还原；
> - [2]中主要也有两种：
>   - 词丢弃噪声：以$p_{wd}$概率独立地丢掉$x_i$，且遍历整个源端；
>   - 重新排列$x$：接受$x'_i$，当其满足条件$\forall i \in \{1,n\},\vert \sigma(i)-i \vert \leq k$;

***[2]. Phrase-Based & Neural Unsupervised Machine Translation, EMNLP 2018, FAIR.***

***[3]. MASS: Masked Sequence to Sequence Pre-training for Language Generation, ICML 2019, MSRA.***

---

---

#### LCA相关实验

1. <del>整理模拟迭代器的代码，与润泽对接，并整合代码；（由润泽搞定）</del>
2. <del>测试代码速度</del>：（暂时待润泽搞定之后）
   - <del>在不同LCA批数据大小的设置下；</del>
   - <del>在关闭grouping循环的设置下；</del>
3. 整理出易用的代码与两位师弟对接：以复现acl short的标准来；



### Mar. 10 2020

> 周二，大部分为讨论.

#### Dymestifying the Learning of UNMT

> **Abstract**
>
> Through experiments we find that independently initializing weights of encoder and decoder can lead to total failure of fune-tuning only with Back-Translation (BT) loss without the help of Denoising Auto-Encoding (DAE) loss. Due to the fact that the learning of UNMT resembles that of Self-Training (ST) or Co-Training (CT), we define a measure named Effective Learning Signal (ELS) based on what the model has learned the right supervision signal from previous self-generated noisy supervised data. We use approximation of this measure along the learning process to quantify the progress that the model has made, which can help explain the success or failure of training.

The learned bilingual knowledge of a model $\mathcal{M}$ from $(x, y)$ can be explained by the source target alignment using attribution algorithm $A$ on a bilingual pair $(x, y)$. This, as a local explanation, reflects how $\mathcal{M}$ makes step-wise prediction of each target token with the most relevant set of source tokens. In our investigation of the learning of UNMT, we try to quantify the progressive bilingual knowledge the model has learned during the standard learning protocol and understand the failure of training without DAE loss.

> **Definition (Correct Supervision Signal)**
>
> Suppose $(x, \hat{y})$ is a BT pair generated by the model given monolingual data $x$, we define that the training pair $(x, \hat{y})$ contains correct supervision if it contains word pairs $x_i, \hat{y}_j$  which can be the potential translation of each other. 

**Note** Since in standard UNMT setting, we do not have access to any bilingual corpus, so in our investigation, we suppose their is a latent bi-text $(x, y) \in \mathcal{B}$, which can be used to induce the alignment $a^{x, y}$ and for training UNMT model as well. When using $\mathcal{B}$ for training the model, we split $\mathcal{B}$ into two half and remove the source and target sentence respectively to create monolingual corpora $\mathcal{D}^{e}$ and $\mathcal{D}^f$. Under this setting, we can decide whether the self-generated data $(x, \hat{y})$ has correct supervision.

- **How to induce alignment $a^{x, y}$?**
  - Here we use two ways to induce the alignment through the whole corpus $\mathcal{B}$:
    - ($\mathcal{M}$-agnostic) We use GIZA++ (which is based on statistical alignment model, i.e. the IBM model) to induce $a^{x, y}$. We **symmetrize** the alignment using both forward and backward modelling.
    - ($\mathcal{M}$-aware) We use $m$ independently and well trained unsupervised NMT model $\mathcal{M}^i$ for attribution, and use their average voting of the top-$k$ salient alignments to get the alignments $a^{x, y}$.
      - Note that, different from $\mathcal{M}$-agnostic method, we can get a set of $k$ features that potentially used for predition as the bilingual knowledge of the model; and we do not symmetrize the alignment.

> **Definition (Effective Supervision Signal)**
>
> Given the generated bitext $(x, \hat{y})$ for BT training ($\hat{y} \rightarrow x$) at step $t$ with model $\mathcal{M}_t$, where $\hat{y} = \mathcal{M}_t (x)$, we define among the Correct Supervision Signal in $(x, \hat{y})$ the Effective Supervision Signal for model at $\mathcal{M}_{t'}$ as the pairs $x_i, \hat{y}_j$, such that through certain attribution algorithm $A$, we can have $\hat{y}_j \in A[x_i; \mathcal{M}_t, k]$, where $A[x_i ; \mathcal{M}_t, k]$ returns the $k$-most salient features for the model to decide the prediction of $x_i$.

The definition is simply saying that the model later on (at step $t'$) are learning the potential Correct Supervision Signal generated previously during BT training. Since for each training example, $\hat{y} \rightarrow x$, we have  where the target $x$ is always the same (while $\hat{y}$ changes during training), we have the same number of Supervision Signal, i.e. $\vert x \vert$, in the sense of word alignment for each $x_i$, namely some are noisy, some are correct and a few are effective. So for each training instance $x$, we see its Correct and Effective Supervision Signal as two random variables during training, which are two sets of word pairs $(x_i, \hat{y}_j)$, denoted as $S^{css}_{x}$ and $S^{ess}_{x}$.

> **Definition (Correct/Effective Supervision Signal Count)**
>
> $\vert S^{css} \vert$ are the count of Correct Supervision Signal, and $\vert S^{ess} \vert$ is the count of effective learning signal.









### Mar. 12 2020

> Little Sunny Thursday.

---

1. Know about GIZA++ and how to use it for obtaining the alignment of each $(x, y)$;











