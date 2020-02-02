---
layout: post
title: "Jan. 1-5"
author: Guanlin Li
tag: log
---

* [Jan. 1 - Wednesday](#jan-01-wednesday)
* [Jan. 2 - Thursday](#jan-02-thursday)
  * [A course on Trustworthy ML at UCB](#a-course-on-trustworthy-ml-at-ucb)

---

#### Jan 01 Wednesday

[1:00] Configure `env` on `jizhi` cluster.

[0:30] Adapt the code and run risk estimation for barrier words.



**Configure env.**

```bash
# create a new env. with conda
conda create -n torch1.0 python3.6

pip install torch
pip install ipdb
pip install ipython
pip install tqdm

# current jizhi server
IP: 9.91.7.206
PW: zyrRjan@EKBwx1
```

`fairseq-epsilon` can run on `torch1.3`.

---

#### Jan 02 Thursday

[1:00] 



##### A course on Trustworthy ML at UCB

- Course format: weekly lecture (60min) + in-depth discussion (20min)
  - Research papers on week's topics: main reading (*must read*) vs background reading.
    - Discussion: what you want to learn from the speaker! <u>Submit questions on Piazza</u>, and use the questions to spark discussion in class.
    - Project:
      1. In-depth literature review (distill template)
      2. Reimplementation + open sourcing them (cf. ICLR repo challenge)
      3. **Conference quality** research project
    - Project schedule
      - 1-month for project proposal
      - 1-month for project milestone report
      - 1-month for poster presentation

- Course overview:
  - Many ways for ML systems to go wrong:
    - input can change unexpectedly, or subtly over time
    - sensors can fail
    - test cases involve novel classes unseen during training
    - adversarials try to hack the system
  - Make sure ML systems do what people expect, avoid silent/unexpected/extreme failures
  - Particularly important in ML: correlated failures
    - Distribution shift
      - Change of user search question (Flu prediction) leads to under-prediction
    - Exploitability
    - Calibration
    - Fairness
      - Minority subpopulations
      - Bias in data
    - Causality: health care, economics; what is the effect of intervention/policy X
      - out-of-distribution
    - Privacy v.s. accuracy
      - Differential privacy
  - Alternative metrics
    - accuracy on variety of dev sets (distribution shift)
    - accuracy on sub-populations (fairness)
    - worst-case over nearby points (robustness)
    - accuracy after applying intervention (causality)
  - Model mis-specification
    - Supervised learning: (wish train=test), model mis-specification doesn't matter.
    - Everything else: mis-specification matters!
      - Mis-specification examples: different distributions *conflict*
  - Aspects of ML works
    - Engineering: build systems that work
    - Science: understand why they work
    - Concepts: mental frameworks for designing/understanding systems
    - Math: formal underpinnings of the above
  - Recommended readings:
    - [Two High-Stakes Challenge in ML](https://icml.cc/2015/invited/LeonBottouICML2015.pdf), Leo Bottou.
    - [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565).
    - [Reflections on Random Kitchen Sinks](http://www.argmin.net/2017/12/05/kitchen-sinks/), Ben Recht