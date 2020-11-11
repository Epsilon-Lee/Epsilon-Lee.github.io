---
layout: post
title: "Research Diary from April 20"
author: Guanlin Li
tag: diary
---

- toc
{:toc}


### April 20

> It's likely a rainy Monday. And at office, the air conditioner is on.

#### Metric-stability of `genbar` detection algorithm

Considered metrics:

- NIST: use `nltk.translate.nist_score`
- WER: use `jiwer` [here](https://pypi.org/project/jiwer/)
- METEOR: use `nltk.translate.meteor_score`
- chrF: use `nltk.translate.chrf_score`
- BERT_score: use [python function](https://github.com/Tiiiger/bert_score#python-function) here

Test their usability.

```python
# `ref` and `pred` are the reference and prediction string respectively
# for example, ref = 'i have a dream', pred = 'i had a dream'

# WER Score
from jiwer import wer
wer_s = wer(ref, pred)  # wer_s the lower the better


# NIST Score
import nltk.translate.nist_score as nist_score
nist_s = nist_score.sentence_nist([ref.split()], pred.split())  # default 5-gram statistics


# METEOR Score
import nltk.translate.meteor_score as meteor_score
meteor_s = meteor_score.single_meteor_score(ref, pred)  # between [0, 1], the larger the better, using WordNet for synonym replacement


# chrF Score
import nltk.translate.chrf_score as chrf_score
chrf_s = chrf_score.sentence_chrf(ref.split(), pred.split())


# BERT Score
from bert_score import score

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

(P, R, F), hashname = score(cands, refs, lang='en', return_hash=True)
print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}')
```



### Apr. 21

> It's Tuesday, and largely sunny outside. The misty whether is coming.

