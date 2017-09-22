---
layout: post
title: "Variational Posterior Estimation via GANs: Bayesian Logistic Regression As Showcase"
author: Guanlin Li
tag: blog
---

> I will start concise presentations of works done by others through exploratory, experimentary practice. This blog is a based on a start of **Ferenc Huszar**'s series [blogs](http://www.inference.vc/variational-inference-with-implicit-probabilistic-models-part-1-2/). His blogs continually demonstrates the use of GANs as a learning mechanism for implicit generative models, such as a variational distribution defined implicitly by a neural network for easy sampling and posterior inference. Personally, I want to thank Ferenc a lot, since his blogs is my source of intuitions on the way to learn and interpret probabilistic generitive modelling, variational methods and GANs. I wish to see his continual updates of his Blog. 

### Exploratory Problem

The problem is kept simple and concise for exploration as a beginner. It is **Bayesian Logistic Regression (BLR)**, the traditional logistic regression formulation augmented with prior on weight $$w$$. 

So let us first restate the Bayesian formulation of logistic regression. **BLR** can be depicted as the above graphical model. $$x_n$$ is the feature vector of a sample point, $$y_n$$ is its corresponding label, $$y_n \in \{0,1\}$$. For the convenience of visualization, I will set that $$x_n \in \mathbb{R^2}$$. For every sample point, the probability of its label is defined as: 


$$ P(y_n \vert w; x_n, b) = p_i^{1[y_n=1]} (1-p_i)^{1[y_n=0]} $$


$$1[\cdot]$$ is the indicator function, where


$$ p_i = \frac{1}{1 + \exp(-b - w^T x_n)} $$

$$p_i​$$ is the logistic sigmod function value of $$(-b-\mathbb{w}^T x_n)​$$. To clarify notation, $$P(\cdot)​$$ is used as a probability (measure) operator, the vertical bar $$ \vert ​$$ means that the expression before is conditioned on the after; and the colon ';' means expressions after are seen as constant. See the following comment. 

> **Comment** More words on the notations. In the book Pattern Recognition and Machine Learning (PRML), every expressions which have the semantic meaning 'observed'/'given'/'conditioned on' are after vertical bar $$\vert$$. E.g. $$ p(t\vert\pi, \mathbb{\mu_1}, \mathbb{\mu_2}, \mathbb{\Sigma}) $$, here we don't know whether $$ \pi, \mathbb{\mu_1}, \mathbb{\mu_2}, \mathbb{\Sigma} $$ are seen as random variables or not as within Bayesian context. In the book Pattern Classification, the notation is consistent with PRML. In Machine Learning: A Probabilistic Perspective (MLaPP), the definition of logistic regression model is: $$ p(y \vert x, \mathbb{w}) = Ber(y \vert sigm(\mathbb{w}^T x)) $$, whereas in PRML is: $$ p(t \vert \mathbb{w}) = y_n^{t_n} \{ 1-y_n \}^{1-t_n} $$, where $$ y_n = p(\mathcal{C}_1 \vert \phi_n)$$ and $$ \phi_n = \phi(x_n) $$. As for MLaPP, it is not clear whether $ x $ is seen as a random variable or not, and both books tacle with Bayesian approach towards logistic regression later on. To be more clarified, I would like to propose the use of colon $ ; $ symbol, and the conditional probability will always have the following form: $$ P(\cdot \vert \cdot ; \cdot) $$. We always see expressions after $$\vert$$ and before ; as r.v.s and expression after ; as constants. This leads to the expression in our first equation, i.e. $$ P(y_n \vert \mathbb{w}; x_n, b) $$. And in $$ \mathcal{N}(x; \mathbb{\mu}, \mathbb{\Sigma}) $$, $$ \mathbb{\mu} $$ and $$ \mathbb{\Sigma} $$ are not seen as r.v.s.


The so-called Bayesian is that we put prior knowledge on the value of $$w$$. Specifically, we assume a Bivariate Gaussian distribution: 



$$P(w) = \mathcal{N}(0, \sigma^2 \mathbb{I}_2) = \frac{1}{2 \pi} \frac{1}{\sigma} \exp \{ - \frac{\vert \vert x \vert \vert^2}{2\sigma^2} \}$$



Since we know Multivariate Gaussian is: 



$$ \mathcal{N}(x;\mathbb{\mu}, \mathbb{\Sigma}) = \frac{1}{(2\pi)^{D/2}} \frac{1}{\vert \mathbb{\Sigma} \vert ^{1/2}} exp\{ -\frac{1}{2} (x - \mathbb{\mu})^T \Sigma^{-1} (x - \mathbb{\mu}) \} $$



Here, the assumed distribution on $$ \mathbb{w} $$ is **spherical Gaussian** with diagnal covariance matrix. We can go on to write down the probability of the data set $$ \mathcal{D} $$ given we know the $$ \mathbb{w} $$, that is $$ P(\mathcal{D} \vert \mathbb{w}; b) $$. And $$ \mathcal{D} $$ stands for the $$ \{ (x_n, y_n)_{n=1}^{N} \} $$. We have: 



$$ P(\mathcal{D} \vert w; b) = P(\{y_n\}_{n=1}^{N} \vert w; \{x_n\}_{n=1}^{N}, b) $$



This is called the likelihood of the data under the model parameterized by $$ \mathbb{w} $$. We want to perform posterior inference on this parameter, that is to derive a computational form of $$ P(\mathbb{w}  \vert \mathcal{D} ) $$. We know posterior estimation is not point estimation but density/distribution estimation. We use Bayes formula to get: 



$$ P(\mathbb{w} \vert \mathcal{D}) = \frac{ P(\mathcal{D} \vert w) P(w)}{ P(\mathcal{D}) } = \frac{ P(\mathcal{D} \vert w) P(w)}{ \int_{\mathbb{w}} d\mathbb{w} P(\mathcal{D} \vert w) P(w) } $$



Very different from Maximum Likelihood Estimation (MLE), to which the computational issue is optimization (maximazing log-likelihood function of data), Bayesian inference or posterior estimation is solving an intractable intergal. Traditional ways of preventing intractability is to restrict prior and posterior to be conjugated, preserve exactability whereas introduce limitations of expressibility. In this blog, we resort to one popular approximation method - Variational Inference which uses a tractable (easy to sample from), parameterized distribution to approximate the real one by minimizing their KL divergence: 



$$ KL(Q \vert \vert P) = \int_{\mathbb{w}} Q log \frac{Q}{P} $$



where $$ P $$ stands for  $$ P(w  \vert\mathcal{D}) $$. In next section, I will derive a form of this KL divergence and show how to parameterize $$ Q $$ so we can simultaneously minimize KL and preserve expressibility of $$ Q $$. 

Before diving into math, I have confused with another question about this variational inference objective - KL divergence. 

> Since we know KL divergence is not symmetric for the two compared distributions, so **WHY** use $$ KL(Q \vert \vert P) $$ instead of $$ KL(P \vert \vert Q) $$?

### Math Derivation

Let us now do some deduction on the KL divergence formula and see what we could get. And to understand why should we use $$Q \vert \vert P$$ instead of $$ P \vert \vert Q $$. In variational inference, we usually parameterize distribution Q, here the only assumption is we use parametric method instead of non-parametric to model Q and the parameters of Q are denoted as $$ \theta_Q $$. So we can make the parameter $$ \theta_Q $$ explicit in the KL term, i.e. $$ KL(Q \vert\vert P; \theta_Q) $$. 



Now, let us try to interpret the objective function. It has two terms: **a).** KL divergence between approximation $$ Q $$ and prior $$ P(\mathbb{w}) $$, which is a spherical Gaussian. **b).** The **EXPECTED** negative <u>log-likelihood of the data set</u> w.r.t. to $$ \mathbb{w} $$ sampled from the approximation distribution. To minimize this $$ \mathcal{O}(\theta_Q) $$ is to find an approximation distribution Q on $$ \mathbb{w} $$ which maximize the expected likelihood while does not make Q too far from the prior (our prior knowledge as a regularization). 

So how to solve this optimization problem? Let us use the simple but powerful gradient descent (GD), unless we can analytically derive the solution by computing the gradient and set it to 0. 

Traditionally, Q is defined within a tractable family of distribution, e.g. exponential family, mixture of Gaussians etc. (TO-DO: derive and implement a traditional version). The reason for restricting Q is: 1). Easily sample from Q, so we can ease this optimization. That is, we can easily compute approximate gradient by monte carlo methods and do GD. 2). [...]. 

> Here I have a second confusion! Even if we have a easily-sample-from approximation distributioin, we must use statistical simulation methods to get sample from. Or even if the sample generator is a Neural Network as in GANs, we must as well sample from a uniform distribution as latent code (input/initialization) to the Neural Net. 
>
> 1. Is this statement true?
> 2. If true,how does this simulation methods actually implemented? [TO-DO: understand VI in Bayesian mixture of Gaussian, see Blei's note]

As a fact, easing sampling by limiting expressibility of Q can result in poor approximations. Can we preserve both easy sampling and expressibility? Yes, we can use ideas from GANs to learn <a href="https://arxiv.org/abs/1610.03483">implicit generative model</a>! The relaxation from explicit Q to implicit Q is remarkable. The difference between explicit and implicit is whether we can directly plug in one sample and calculate its prob. density. In implicit models, we cannot. 

In the following sub section, I will present the way to transform  $ \mathcal{O}(\theta_Q) $ to suitable forms for GAN-like training.

## Reduction

The motivation all starts from our willing to expressing approximation distribution Q implicitly, here by using a Feed Forword neural network with initialization random samples from a uniform distribution. 

We denote the implicit generative model $$ FFNN(\cdot) $$, and weight $$ \mathbb{w} $$ for the BLR is sampled from this FFNN, i.e. $$ \mathbb{w} \thicksim FFNN(z), z \thicksim U(\cdot) $$, where $$ z $$ is sampled from uniform distribution $$ U(\cdot) $$. Within the framework of GANs, we also call the FFNN as a generator D. 

Now, we have a random sample generator which is untrained! Sampling from it is as simple as generating uniformly distributed samples. Let us see how this can help compute the objective function $$ \mathcal{O} $$. (We denote parameters of this FFNN as $$ \theta_G \in \theta_Q $$) I rewrite the objective below: 



$$ \mathcal{O(\theta_Q)} = \mathbb{E}_{ \mathbb{w} \thicksim Q(\mathbb{w}) } log \frac{ Q(\mathbb{w}) }{ P(\mathbb{w}) } - \mathbb{E}_{ \mathbb{w} \thicksim Q(\mathbb{w}) } { log P(\mathcal{D} \vert \mathbb{w}) } $$



There are two terms in this objective: 

1. **Expectation of log prob. ratio w.r.t. $$ \mathbb{w} \thicksim Q $$.** Since Q is parameterized as FFNN, drawing samples of w is easy. Suppose we draw some samples as $$ \{ \mathbb{w}_i \}_{i=1}^{M} $$, monte carlo methods can be used as an estimate of this expectation, as well its gradient. For every $ \mathbb{w}_i $, $$ P(\mathbb{w_i}) $$ is easy to compute, but we do not have explicit prob. value of $$ Q(\mathbb{w}) $$. <u>So this is a issue to work out!</u> 
2. **Expectation of log evidence w.r.t. $$ \mathbb{w} \thicksim Q $$.** As stated in 1, easy to get samples. The term, $$ log P(\mathcal{D} \vert \mathbb{w}) = log \Pi P(y_i \vert \mathbb{w}; x_i, b) = \Sigma log p_i^{\mathbb{1}[y_i=1]} (1 - p_i)^{\mathbb{1}[y_i=0]} $$, is trivial to compute as well (unless number of data points is huge!). 

So we should only solve the computation of prob. ratio $$ \frac{ Q(\mathbb{w}) }{ P(\mathbb{w}) } $$. 

It is **very creative** (from my current viewpoint) to think of the computation of this density ratio of two distributions within the framework of classification. That is, we can unify two densities under one classification problem, i.e. **whether the sample x is from Q or P**. Now, let us derive the classification approach to density ratio estimation. 

Assume that $$ P(\mathbb{w}, l) $$ is a joint probability over $ \mathbb{w} $ and $$ l \in \{ -1, 1 \} $$, a class label (indicator of which underlying distribution $ \mathbb{w} $ comes from). We can factorize $$ P(\mathbb{w}, l) $$ in two ways: 



$$ P(\mathbb{w}, l) = P(\mathbb{w} \vert l) P(l) = P(l \vert \mathbb{w}) P(\mathbb{w}) $$



Here, the conditional probability of $$ \mathbb{w} $$ given label $$ l $$ means P or Q when $$ l $$ is 1 or -1. So for each value of $$ l $$, we can write down two equations: 



$$ P(\mathbb{w} \vert l=1) P(l = 1) = P(l=1 \vert \mathbb{w}) P(\mathbb{w}) $$

$$ P(\mathbb{w} \vert l=-1) P(l = -1) = P(l=-1 \vert \mathbb{w}) P(\mathbb{w})$$



Since we want to compute Q/P, so we can divide **lhs** and **rhs** of these two equations respectively and get: 



$$ \frac{P(\mathbb{w} \vert l=1)}{P(\mathbb{w} \vert l=-1)} \cdot \frac{P(l=1)}{P(l=-1)} = \frac{P(l=1 \vert \mathbb{w})}{P(l=-1 \vert \mathbb{w})} $$



We know so: 



$$ \frac{Q(\mathbb{w})}{P(\mathbb{w})} = \frac{P(l=-1)}{P(l=1)} \cdot \frac{P(l=1 \vert \mathbb{w})}{P(l=-1 \vert \mathbb{w})}  $$



Most of time, we don't know whose sample frequency is larger, so we assume equal frequency of choosing Q and P for $ \mathbb{w} $'s generation. This result in: $$ P(l=1) = P(l=-1) = \frac{1}{2} $$. And moreover: 



$$\begin{align}
\frac{Q(\mathbb{w})}{P(\mathbb{w})} 
&= \frac{P(l=1 | \mathbb{w})}{P(l=-1 | \mathbb{w})} \\
&= \frac{P(l=1 | \mathbb{w})}{1 - P(l=1 | \mathbb{w})}
\end{align}$$



Then we can use another model to parameterize $$ P(l=1 \vert \mathbb{w}) $$, and ensure that it can really discriminate between $ \mathbb{w} $ sampled from Q and P. More specific, when given samples sampled from $$ Q $$, the discriminator should output a probability close to 1, otherwise the probability should be close to 0. 

This leads to our design of a discriminator D parameterized by $$\theta_D \in \theta_Q $$ and another objective, with learning which can ensure the approximate accuracy of estimating $$ P(l=1 \vert \mathbb{w}) $$. 

This leads to our design of a discriminator D parameterized by $$P(l=1\vert \mathbb{w})$$, and another objective, with learning which can ensure the approximate accuracy of estimating. Specifically, we can use another FFNN with parameter $$ \theta_D $$ to specify the discriminator. That is what we will do in implementation. 

Till now, we have finished parameterizing our learning objective $$ \mathcal{O}(\theta_Q) $$ by: **a).** an implicit generative model G to sample $$ \mathbb{w} $$, **b).** a discriminator D with an auxiliary objective to help density ratio estimation (we use G to substitute FFNN): 

$$
\begin{align}
      \mathcal{O}(\theta_Q) &= \mathbb{E}_{\mathbb{w} \thicksim G} log \frac{D(\mathbb{w})}{1 - D(\mathbb{w})} + \mathbb{E}_{\mathbb{w} \thicksim G} log P(\mathcal{D} | \mathbb{w}) \\
                            &= \mathbb{E}_{z \thicksim U(\cdot)} log \frac{D(G(z))}{1 - D(G(z))} + \mathbb{E}_{z \thicksim U(\cdot)} log P(\mathcal{D} | G(z)) \\
\end{align}
$$


$$ \text{Auxiliary}: \mathcal{O}(\theta_D) = \mathbb{E}_{\mathbb{w} \thicksim P(\mathbb{w})} log D(\mathbb{w}) + \mathbb{E}_{z \thicksim U(\cdot)} log (1 - D(G(z))) \ $$



> <b>Comment.</b> I wonder whether my derivation is correct, since it is different from Ferenc's. I took a look into Ferenc's <a href="https://gist.github.com/fhuszar/a597906e994523a345744dc226f48f2d">ipython Notebook implementation</a> and found that the code is according to his definition of the discriminator's loss. I am still working on an explanation. 

