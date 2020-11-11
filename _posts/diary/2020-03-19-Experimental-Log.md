---
layout: post
title: "Experimental logs from 3/19/2020 - 3/20/2020"
author: Guanlin Li
tag: diary
---



- toc
{:toc}




### Mar. 19

> It's cloudy outside. Back to the work building from quarantine, and happy to see her here.

1. 完成`PD`的代码与调试；
2. 在clean数据上跑fast_text与GIZA++；

---

#### `masked_select`

```python
alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
pred_mask = alen[:, None] < len2[None] - 1
y2 = x2[1:].masked_select(pred_mask[:-1])
```



#### `GIZA++`

1. 使用`plain2snt.out`处理双语数据：

   ```bash
   ~/Software/giza/plain2snt.out en.txt fr.txt
   # output
   en_fr.snt
   fr_en.snt
   en.vcb
   fr.vcb
   ```

2. 使用`snt2cooc.out`生成共现文件：

   ```bash
   ~/Software/giza/snt2cooc.out fr.vcb en.vcb fr_en.snt > fr_en.cooc
   ~/Software/giza/snt2cooc.out en.vcb fr.vcb en_fr.snt > en_fr.cooc
   ```

3. 生成词类：

   ```bash
   ~/Software/giza/mkcls -pen.txt -Ven.vcb.classes opt
   ~/Software/giza/mkcls -pfr.txt -Vfr.vcb.classes opt
   ```

4. 运行GIZA++：

   ```bash
   mkdir -p e2f f2e
   
   # En-Fr
   ~/Software/giza/GIZA++ -S en.vcb -T fr.vcb -C en_fr.snt -CoocurrenceFile en_fr.cooc -o e2f -OutputPath e2f
   
   # Fr-En
   ~/Software/giza/GIZA++ -S fr.vcb -T en.vcb -C fr_en.snt -CoocurrenceFile fr_en.cooc -o f2e -OutputPath f2e
   ```

5. 抽取词对齐：

   ```bash
   python align_sym.py e2f.A3.final f2e.A3.final > aligned.grow-diag-final-and
   ```

   















