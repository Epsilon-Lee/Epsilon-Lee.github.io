---
layout: post
title: "On Researhch in Machine Learning and Natural Language Processing"
author: Guanlin Li
tag: blog
---

* [Research Blueprint](#research-blueprint)
* [Develop taste](#develop-taste)
* [Conduct research scientifically](#conduct-research-scientifically)

The motivation of this blog post is from recent discussion on Twitter, where researchers share their experience and taste of research.

- [1] [An Opinionated Guide to ML Research](http://joschu.net/blog/opinionated-guide-ml-research.html), John Schulman, 2020/01/24.

But the most original stimulus is Jacob and Lipton's paper:

- [2] [Troubling Trends in Macine Learning](http://approximatelycorrect.com/2018/07/10/troubling-trends-in-machine-learning-scholarship/), Zachary Lipton and Jacob Steinhardt, around ICML 2018.

and a not so well-known piece of research process abstraction by Jacob:

- [3] [Research as a Stochastic Decision Process](https://www.stat.berkeley.edu/~jsteinhardt/), Dec. 2018, which you can find at his home page.

I want to *highlight* their main points and to *absorb* them into my own research taste, as I am special in Natural Language Processing (NLP) or Computational Linguistics (CL), which may be (quite) different from ML research.

> This is blog post should not be treated as a methodology discussion on how to do tasteful or influential research, but a retrospect of my own research attitude, which might not be personalized to others who are doomed to have different research experiene.

---

#### Research Blueprint

[1] is written by [John Schulman](http://joschu.net/index.html) who is a research scientist at OpenAI and his work on policy optimization or policy gradient, e.g. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) is the one of the most well-known research direction in deep reinforcement learning. His blog post mainly talks about how to conduct research in ML. The article's main structure is:

- How to choose problem?

  - Understand, compare and criticise what makes a work influential or obscure (obsolete)?

    - (**Utility in practice**) Is it simple to xxx but very practical? (`xxx = derive, understand, implement`)
    - (**Utility in theory**) Is the theory very ground-breaking, game-changing?
    - (**Utility in inspiring others**) Is the observation or understanding very profound? Or are the insights really motivating?

    > John advocates goal-driven research instead of idea-driven research, since goal-driven research is the one that you can continually dedicate yourself into.
    >
    > In my opinion, idea-driven research comes like some hunches which strike on you randomly but causally due to the recent papers you are reading. It is like a <u>brick</u> in a research program or project.
    >
    > > **John's definition of idea-driven research**
    > >
    > > *"Follow some sector of the literature. As you read a paper showing how to do X, you have an idea of how to do X even. better. Then you embark on a project to test your idea."*
    >
    > But goal-driven research is more like a programme which you conduct and insist on in a long run and may finally make certain <u>unignorable</u> contribution to the community that identifies yourself with what you have done.
    >
    > > **John's definition of goal-driven research**
    > >
    > > *"Develop a vision of some new capabilities you'd like to achieve, and solve problems that bring you closer to that goal. In your experimentation, you test a variety of existing methods from the literature, and then you develop your own methods that improve on them."*

- How to make continuous progress on the problem?

  1. **Keep a notebook** where to record your ideas, observations about the field its super and sub fields. The following excerpts are really motivating.

     > *"I have done this through 5 years of grad school and 2 years at OpenAI"*
     >
     > *"I create an entry for each day. In this entry, I write down what I'm doing, ideas I have, and experimental results"*
     >
     > *"Every 1 or 2 weeks, I do a review, where I read all of my daily entries and I condense the information into a summary"*
     >
     > *"Usually, my review contains sections for experimental findings, insights, code progress, and next steps/future work."*
     >
     > *"I often look at the previous week to see if I followed up on everything I thought of that week."*

  2. **Switch problem less often**  Stick to write a paper, and list the unsolved research questions of the paper, which always motivate better formulation of research questions for you to work upon.

     > *"spend one day per week on something totally different from your main project. This would constitue a kind of  epsilon-greedy exploration, and it would also help to broaden your knowledge."*

- and How to improve yourself generally during the above procedures?

  - Read fundamental **books**,  e.g. PRML, [ITILA](http://www.inference.org.uk/mackay/itila/), Elements of Information Theory, Numerical Optimization, [Elements of Causal Inference](http://web.math.ku.dk/~peters/elements.html).
  - Continuosly enrich your toolkit of problem solving and manage your knowledge as an *independent* researcher;
  - As for me, e.g. normalizing flow in deep generative models, generalization theory of neural nets, influence functions, invertible neural models and differential equation's perspective of neural nets, neural tangant kernels and kernel mean embedding of distribution, non-convex optimisation, the argument about Bayesian Deep Learning and uncertainty estimation or calibration, mutual information estimation etc.

> Above is a summary of [1], which is very handy.

---

#### Develop Taste

[2] is my favorite paper till now during my PhD persuit, since it gives me standard or`dos and donts` when look into others or my own research.



---

#### Conduct Research Scientifically



