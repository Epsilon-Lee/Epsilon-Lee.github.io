- **[V1]** Take top 20,000 frequent token and prune the rest;
- **[V2]** Constrained decoding introduced in [mBART](https://arxiv.org/abs/2001.08210) [1]. HOW to get the independent vocabulary?

Given two vocab sets $A$, $B$, we construct new $A'$ and $B'$ as follows (take $A'$ for example).

1. For every token $w \in A$ if, $w \in B$, check that:
   - $freq_A(w) / freq_B(w) < r$, where $r$ is a usage ratio; small $r$ indicating that the token $w$ is more frequently used in language $B$ instead of language $A$, so we probably encounter a foreign word for $A$;
2. If $freq_A(w) / freq_B(w) < r$, then remove $w$ from $A$;
3. Iterate all $w \in A$, return the final $A$ as $A'$;

[1]. [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210), FAIR 2020.



一些可以用来提供训练过程描述与理解的**任务**或**统计量**

- 基于Force Decoding的统计量
- 基于Real Decoding的统计量



设无监督模型$$\mathcal{M}$$在训练语料$$\mathcal{D}$$上进行标准协议的训练，并记$$\mathcal{D}^b$$为单语语料$$\mathcal{D}$$对应的双语语料，为了让训练的动态过程最大程度地仅仅受到两个损失函数$l_{DAE}$与$l_{BT}$的影响.

在$\mathcal{D}$上训练的双语语言模型为$$\mathcal{M}_{lm}$$，在双语语料$$\mathcal{D}^b$$上训练的、仅以$$\mathcal{D}$$作为目标的翻译模型为$$\mathcal{M}_{mt}$$.

若我们希望度量：翻译$$(x, y)$$句对时，$$t$$位置语符的分布受到源端$$x$$的影响有多大，可以用下面量进行度量：

- 即我们假设$$y_{<t}$$与$$x$$对$$t$$位置语符分布是线性可加的，那么我们定义$$x$$对$$y_t$$的影响为：
  - $$P(y_t \vert x, y_{<t}) - P(y_t \vert y_{<t})$$
  - 但其实真正的贝叶斯公式下的等式为：
    - $$P(y_t \vert x, y_{<t}) = P()$$



衡量DAE/BT起到训练模型翻译能力的作用还是语言建模的作用，其实类似于衡量构建DAE/BT损失的数据对更像翻译的数据还是更像语言模型的数据.





