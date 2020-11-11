### 无监督与半监督神经机器翻译的变分解释



- toc
{:toc}




无监督与半监督神经机器翻译均可视为最大化下面的边际似然.

- 单语数据上的边际似然概率：$$P(y) = \sum_{x} P(x, y)$$
- 我们对其进行变分处理

$$
\begin{align}
\log P(y) &= \log \sum_{x} P(x, y) = \log \sum_{x} P(x \vert y)/P(x \vert y) \cdot P(x, y) \\
&= \log \mathbb{E}_{P(x \vert y)} P(x, y)/P(x \vert y) \\
&\geq \mathbb{E}_{P(x \vert y)} \log [ P(y \vert x) \cdot \frac{P(x)}{P(x \vert y)} ] \\
&= \mathbb{E}_{P(x \vert y)} \log P(y \vert x) - \mathbb{E}_{P(x \vert y)} \log \frac{P(x \vert y)}{P(x)} \\
&= \mathbb{E}_{P(x \vert y)} \log P(y \vert x) - \text{KL}[P(x \vert y); P(x)]
\end{align}
$$

- 从损失下降的角度，我们要最小化：$$-\log P(x)$$，则要最小化其上界，由两项组成：

  - 重建负对数似然项：$$\mathbb{E}_{P(x \vert y)} - \log P(y \vert x)$$

  - KL散度项：$$\text{KL}[P(x \vert y); P(x)]$$

    > 即我们期望使得这两项都尽可能小.
    
    - 另一方面，我们可以进一步将KL散度项拆分为：
    
      - $$\mathbb{E}_{P(x \vert y)} \log P(x \vert y)$$：该项为反向翻译模型$$P(x \vert y)$$的**负**熵
    
      - $$- \mathbb{E}_{P(x \vert y)} \log P(x)$$：该项为反向翻译模型$$P(x \vert y)$$与语言模型$$P(x)$$的交叉熵
    
        > 即最小化KL距离的作用从信息论的角度有两个：一是最大化首分布的熵，让其有能力包含充分的信息；二是最小化与尾分布的交叉熵，让首-尾分布更接近.

---

#### 无监督神经机器翻译

**DAE的正则作用.** 无监督翻译中DAE损失的一方面作用即是使KL散度项不至于太大，从而负对数边际似然$$- \log P(y)$$不至于有一个特别大的上界.

---

#### 半监督神经机器翻译

上述变分形式中的$$P(x \vert y)$$通过反向翻译模型$$P_\theta(x \vert y)$$进行近似，即该模型在一定数量的双语数据上进行有监督训练.





#### 实验一

算法实现

```
1. 在原始的y上，由 P(x|y) 采样 k个batch，x'
2. 根据采样得到的x，计算 - log P(y|x')，得到负对数损失
3. 计算P(x'|y)，计算P(x')，均为句子上（非词符上）的概率，并计算 ratio = P(x'|y) / P(x)，
```



---

$$
\begin{align}
& \nabla_\theta \mathbb{E}_{q_\theta(x \vert y)} [\log q_\theta(x \vert y) - \log p(x)] \\
= & \nabla_\theta \int_x q_\theta(x \vert y) \log q_\theta(x \vert y) - \nabla_\theta \int_x q_\theta(x \vert y) \log p(x) \\
= & \int_x \log q_\theta \nabla_\theta q_\theta + \int_x q_\theta \nabla_\theta \log q_\theta - \int_x \log p(x) \nabla_\theta q_\theta \\
= & \mathbb{E}_q \log q_\theta \nabla_\theta \log q_\theta + \mathbb{E}_q \nabla_\theta \log q_\theta - \mathbb{E}_q \log p(x) \nabla_\theta \log q_\theta \\
\approx & \log q_\theta \nabla_\theta \log q_\theta + \nabla_\theta \log q_\theta - \log p(x) \nabla_\theta \log q_\theta \\
= & (\log q_\theta + 1 - \log p(x)) \cdot \nabla_\theta \log q_\theta
\end{align}
$$