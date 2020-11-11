---
layout: post
title: "Numpy and PyTorch Technology"
author: Guanlin Li
tag: notes
---

- toc
{:toc}


### Permute a batch of token id

```python
# x, xlen; x shape [N, L]
x_ = x.clone()
for i in range(x.shape[0]):
    x[i][1:xlen[i] - 1].copy_(permute(x_[i][1:xlen[i] - 1]))
```

