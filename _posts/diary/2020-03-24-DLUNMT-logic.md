## The logic of DLUNMT



- toc
{:toc}


### What is the right logic?

> This section is to clarify why we want to investigate the emergence of correct supervision and learned knowledge during the training of (unsupervised) NMT model.

#### Model's bilingual translation knowledge

What source-side features does the model $\mathcal{M}_t$ depends on, when making decision or predicting $y_t$ given $(x, y_{<t})$? These source-side features represents the model's ***local*** knowledge of the translation task.

> Here, we do not consider the knowledge of making sentential translation smooth and natural (due to the target-side features); since this can be easily learned through MLE on back-translation data with target as natural sentence.

- Given arbitrary sentence pair $(x, y)$ which has a few aligned parts (i.e at least 1 aligned word pair), can the model still attribute the bilingual knowledge right?
  - If $x_i, y_j$ are aligned, can $A^{\mathcal{M}}[y_j] = x_i$ or $A^{\mathcal{M}}[x_i] = y_j$, where $A$ is certain attribution method for locate the most relevant source-side features.
- Given arbitrary sentence pair $(x, y)$ which has no aligned parts, does the most relevant features found by an attribution method mean something?









#### Some research Qs

- When does the model start to provide correct supervision signal itself for its dual model to learn?
- When the dual model actually learns the supervision signal itself?
  - $t$ and $t'$'s:
    - $(\mathcal{M}_t[x], x)$ if contains correct supervision signal, investigate the following two Qs:
      1. Can the dual model at $t$ attribute correctly such bilingual supervision signal?
      2. Can the dual model at $t'$ attribute correctly such bilingual supervision signal after $\delta(t) = t' - t$?
         - How fast? that is $\delta(t)$;
         - Catastrophic forgetting when $t$ attribute it right but $t'$ attribute it wrong;



### Another logic

> This work is dedicated to a deep understanding of the effectiveness of each loss function during standard UNMT training, namely the successive updates with DAE and BT losses. Previous study, mainly uses ablation to show that both losses are of great importance of preventing the model from complete failure of training. Here, we propose ***two*** simple measures and conduct training on pseudo monolingual sentences *with* translations to investigate each loss's effect on the model's learning of bilingual translation knowledge.
>
> Several hypothetical findings are:
>
> - DAE is very important at the initial stage of training which guarantees the model from degeneration to a language model;
> - DAE training alone can make the model be aware of bilingual translation knowledge, further representation probing shows its effect of producing universal representations;
> - BT is very crucial for enhancing the model to perform even better later in training, which is actually the result of model duality and Co-Training.

#### Loss Change Decomposition

In standard UNMT training, each time step $t$ there are two weights updates carried out:

- i) firstly, DAE update;
- ii) then, BT update.

Actually, the order of the updates really do not matter, since we can always shift one update further regroup the order as above.

---

**What if we do real multi-task training**

If we do real multi-task training, that is the gradient of DAE loss and BT loss are evaluated and calculated on same model weight, say $\theta$ and $\delta_{dae}(\theta)$, $\delta_{bt}(\theta)$, and then $\theta$ is updated through $\theta - lr \cdot (\delta_{dae} + \delta_{bt})$, we can use first order approximation as in [1] to decompose the loss change:

$$
\begin{align}
& \mathcal{L}(\theta - lr \cdot (\delta_{dae} + \delta_{bt})) - \mathcal{L}(\theta) \\
= & \mathcal{L}(\theta - lr \cdot (\delta_{dae} + \delta_{bt})) - \mathcal{L}(\theta - lr \cdot \delta_{dae}) + \mathcal{L}(\theta - lr \cdot \delta_{dae}) - \mathcal{L}(\theta) \\
\approx & \nabla \mathcal{L}(\theta - lr \cdot \delta_{dae}) \cdot (-lr \cdot \delta_{bt}) + \nabla \mathcal{L}(\theta) \cdot (-lr \cdot \delta_{dae}) \\
= & \Delta\mathcal{L}_{bt} + \Delta\mathcal{L}_{dae} \\ 
\end{align}
$$

And this derivation is exactly the same in the LCA paper [1].

---

Borrowed from the concept of Loss Change Allocation [1], we allocate the loss degradation of the model on pseudo bilingual data to the update before and after DAE or BT:

- $\Delta \mathcal{L}^t_{dae} = \mathcal{L}(\theta^t - lr \cdot \delta_{dae}) - \mathcal{L}(\theta^t)$

- $\Delta \mathcal{L}^t_{bt} = \mathcal{L}(\theta^t - lr \cdot (\delta_{dae} + \delta_{bt})) - \mathcal{L}(\theta^t - lr \cdot \delta_{dae})$

  > And $\theta^{t+1} = \theta^t - lr \cdot (\delta_{dae} + \delta_{bt})$

We could then compute and draw the moment bilingual loss contribution above in experiments.



##### Implementation Details

The biggest question is: ***where to evaluate the updated model parameter?***

Original update flow at each step $t$:

```python
1. step_(en/fr)   # pick uniformly random
- 1.1 step_en/fr: x1_en, x1_fr - c(x_en) -> x_en
- 1.2 step_fr/en: x2_fr, x2_en - c(x_fr) -> x_fr
    
    The total latent bilingual loss: -log_p(x1_en|x1_fr) + -log_p(x2_fr|x2_en)

2. step_(en->fr/fr->en)  # pick uniformly random
- 2.1 step_(en->fr/fr->en): x1_en, x1_fr_hat, x1_fr
- 2.2 step_(fr->en/en->fr): x2_fr, x2_en_hat, x2_en
```

**Alternative 1: using the batches of the latent bitext at BT step**

- Ignore the direction
- Ignore the latent bitext at DAE step
- Combine the two losses on each latent bitext at DAE/BT update
- Use latent bitext batches only from BT step

```
# write a holistic_step() function
1. Get both DAE's two batch and BT's two batch
2. With theta_0, evaluate loss on the two BT batches: loss_0
3. Do DAE update to get theta_1
4. With theta_1, evaluate loss on the two BT batches: loss_1
5. Do BT  update to get theta_2
6. With theta_2, evaluate loss on the two BT batches: loss_2
```

> *NOTE*
>
> This choice is made based on our focus of online BT to provide correct supervision signal for the dual model to learn from $\mathcal{M}[x] \rightarrow x$, where one of the best correct supervision signals are contained in the latent bitext $y \rightarrow x$.
>
> In fact, we are measuring the highly correlated batches $\mathcal{M}[x] \rightarrow x$ and $y \rightarrow x$ for the BT update but uncorrelated batches $c(x') \rightarrow x$ and $y \rightarrow x$. So when making certain conclusions, we should be careful about the setting for leading to actually trivial findings.
>
> The correlation will lead to findings like when the model become better, it is reasonable that the learning from $\mathcal{M}[x] \rightarrow x$ will transfer will to the degradation of $y \rightarrow x$ due to the highly similarity between $\mathcal{M}[x]$ and $y$, but it is not quite clear how DAE update on an unrelated batch $c(x') \rightarrow x$ will.

**Alternative 3: using arbitrary batch of latent bitext in the training corpus**

- Ignore the direction
- Ignore the latent bitext at DAE step
- Combine the two losses on each latent bitext at DAE/BT update
- Use latent bitext batches for DAE and BT respectively

```
# still write an holistic_step() function
1. Get both DAE's two batch and BT's two batch
2. With theta_0, evaluate loss on the two DAE batches: loss_0
3. Do DAE update to get theta_1
4. With theta_1, evaluate loss on the two DAE batches: loss_1

5. With theta_1, evaluate loss on the two BT  batches: loss_2
5. Do BT  update to get theta_2
6. With theta_2, evaluate loss on the two BT  batches: loss_3
```

> *NOTE*
>
> DAE might be really helpful for learning universal representations, which might come from the learning of $c(x') \rightarrow x$, which could somehow bring representation of similar meaning sentences in different language close to each other and help with the loss.
>
> But the problem is that at each step $t$, the loss degradation for DAE or BT update is not directly operate on the same batch of data, which might be another source of influence on the magnitude of loss change.

**Alternative 2: using another random batch generated from the same train set**

- Ignore the direction
- Ignore the latent bitext at DAE step
- Combine the two losses on each latent bitext at DAE/BT update
- Use the same latent bitext batches for DAE and BT, which come from another iterator

```
# still write an holistic_step() function
1. Get both DAE's two batch and BT's two batch, and one extra batch with a new iterators
2. With theta_0, evaluate loss on the two DAE batches: loss_0
3. Do DAE update to get theta_1
4. With theta_1, evaluate loss on the two DAE batches: loss_1

5. With theta_1, evaluate loss on the two BT  batches: loss_2
5. Do BT  update to get theta_2
6. With theta_2, evaluate loss on the two BT  batches: loss_3
```

**Alternative 4: using a fixed batch**

...



#### Effective Supervision Signal

This section describes a discrete measure, which can shed lights on how each loss helps the model <u>cumulatively</u> learn the supervision signal emerged during online back-translation.

Since at each step $t$, the model first updates on monolingual data $x$ by denoising $c(x)$ for each language, and then updates on back-translation data $\hat{y}$ with $x$ as target.

> **Legancy**
>
> So given a successive sequence of updates interleaved by DAE and BT, we can measure at certain time step $t'$, given access to all previous generated BT corpus $b_1, b_2, \dots, b_{t'}$ that potentially contains bilingual knowledge, how effective is each loss update corresponding to a learned knowledge of word alignment.









