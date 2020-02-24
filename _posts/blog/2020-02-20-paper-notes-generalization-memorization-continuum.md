---
layout: post
title: "Notes on Exploring the Memorization-Generalization Continuum in Deep Learning"
author: Guanlin Li
tag: blog
---

- toc
{:toc}
### Draft notes.

- > ***Definition (Consistency score or C-score).***
  >
  > *The expected accuracy of a <u>particular model architecture</u> trained with a <u>fixed sized training set</u> on a held-out instance.*
  >
  > The formal form of the definition is:
  >
  > $$C(x, y)_{\mathcal{A}, \mathcal{P}, n} = \mathbb{E}_{D \sim_n \mathcal{P}} [\hat{y}_{\mathcal{A}} = y \vert D, x]$$
  - Two derived Qs:

    - **(Estimation problem)** to estimate the score accurately and efficiently?

      - kernel density estimation on <u>input</u> & <u>hidden representation</u>
      - time course of training (learning speed)

    - **(Usage problem)** to utilize the score? 

      $$\Rightarrow$$ ***Debugging dataset***

      > "*we show that the score identifies out-of-distribution and mislabeled examples at one end of the continuum and regular exampels at the other end.*"
      
      - Potential applications:
        - Understand learning (memorization & generalization) dynamics
        - Curriculum learning, out-of-distribution detection
        - Active data collection
    
  - Relationship to Feldman's memorization score:

    > "*defined relative to a dataset that includes $$(x,y)$$ and measures the change in the prediction accuracy on $$x$$ when $$(x, y)$$ is removed from the dataset*".

- **Empirical Estimation of C-Score**

  - In practice, we usually have a fixed data set $$\mathcal{D}$$ consisting of $$N$$ i.i.d. samples from the underlying distribution;

    - Averaging over i.i.d. subsamples of size $$n$$

    - > ***Empirical C-score***
      >
      > $$\hat{C}(x, y)_{\mathcal{A}, D, n} = \hat{\mathbb{E}}^r_{D \sim_n \mathcal{D}\setminus(x,y)} [\text{Pr}(\hat{y}_{\mathcal{A}}=y \vert x, D)]$$

    - To estimate this score, they propose a ***neat*** $$k$$-fold validation procedure, which train the model from scratch for $$k$$ times on an $$n$$-sized

- **Proxies of C-Score**

  - ***Kernel density estimation in input space***

    - $$\hat{C}^{\pm L}(x, y) = \frac{1}{N} \sum^{N}_{i=1} 2 \cdot (\mathbb{I}[y = y_i] - 1/2) \cdot \mathcal{K}(x, x_i)$$

    - $$\hat{C}^{+L} = \frac{1}{N} \sum^{N}_{i=1} \mathbb{I}[y = y_i] \cdot \mathcal{K}(x, x_i)$$

    - $$\hat{C}(x) = \frac{1}{N} \sum^{N}_{i=1} \mathcal{K}(x, x_i)$$

      > ***Questions.***
      >
      > - How to decide the sample size or support $$N$$?

  - ***Kernel density in hidden space***

    - Some with the above but $$x$$ currently means the hidden representations.

      > ***Questions.***
      >
      > - To use which layer's hidden representations? The one before the softmax projection classification layer? Or the logits layer?
      > - Will it make a difference choosing different layers' representations?

  - ***Learning speed***

    - The INTUITION: "*a training example that is consistent with others should be learned quickly because the gradient steps for all consistent examples should be well-aligned*".
    - The authors gives definition and analyses of two quantitative definitions of learning speed.
      - **Moment model score**: at each epoch $$k$$, given the instance $$(x, y)$$ either use i) $$p_y$$ or use ii) $$p_{max}$$ which is the i) model probability of the label given $$x$$ (label-aware) or ii) maximum of the model probability given $$x$$ (label-unaware);
      - **Cumulative model accuracy**: from beginning of training and till epoch $$k$$, the cumulative prediction hit count of $$y = \hat{y}$$ divides $$k$$; the authors call this *cumulative binary training loss*.

---

### Insights

- This work is conducted along a very logical line of research thinkings and questions which can be referred to as a gold guideline of carrying out research:
  - *firstly*, they propose the importance of understanding the learning phenomenon of deep neural networks from ***memorization*** of specific irregular training instances to abstraction (or ***generalization***) of relatively more regular instances. They argue by an example of human learning verb usage with reflections to further instantiate their motivation, which I think is very approriate and persuasive;
  - *secondly*, they give the definition of their proposed instance-wise *C-Score* and initially show some images from ImageNet that are labeled with their C-Score to demonstrate its capability to capture regularities to irregular cases.
  - *thirdly*, they provide approximation of calculating C-Score according to the original definition, to actually calculate it along three reasonable sized image datasets, namely, MNIST, CIFAR-10, CIFAR-100. And they show a nice visualization of a 2-D Countour distribution map on ImageNet which furthermore states the importance of variance of the C-Score in each class: larger the variance is, more diverse the instances that class has.
    - To extend their analyses of the rationale of the C-Score and its approximation, they conduct experiments on histogram (distribution) analysis of C-Score w.r.t. dataset ratio $$s$$ for training model from scratch, integral C-Score which average along different dataset ratios, or integral C-Score changing by grouping instances under C-Score; and rank correlation between point C-Score and integral C-Score;
    - The above experiments give insights on: i) monotonicity nature of C-Score along groups that has similar C-Score when varying dataset ratio; ii) integral C-Score makes instances more identifiable due to less concentration of the C-Score histogram; iii) different datasets should have its own proper $$s$$ to align point C-Score well with integral C-Score.
  - *fourthly*, they also provide three proxies of the C-Score which can be computed within one training run; and *compare them with C-Score under rank correlation*; during the course of comparison, many insights are presented:
    - The importance of knowing label information: since the definition of C-Score uses label information, the proxies who use label information aligns better with C-Score;
    - Input space is not a good representation space compared to hidden representations; and the learning of hidden representations somehow preserve the structure of the label-information, that is, different instances' hiddens cluster far away with different labels but close to each other within the same class; so hiddens do not "discard inter-class relationships";
    - Model scores during course of training show the best correlation with C-Score, especially cumulative binary training loss, which has ~0.87 Spearman's rank correlation.

---

### Some connections

- Instance reweighting
  - [What is the effect of importance weighting in deep learning](http://zacklipton.com/media/papers/importance-weighting-byrd-lipton2019.pdf), ICML 2019.

- Debugging dataset
  - [Influence function](https://arxiv.org/abs/1703.04730) [related works](https://arxiv.org/abs/1905.13289)
  - [Poisoning](https://arxiv.org/abs/1910.14147v1) [datasets](https://arxiv.org/abs/1811.00741)
- Over-fitting, under-fitting under noise
  - See this [post](https://epsilon-lee.github.io/blog/on-learning-under-dataset-noise/)





















