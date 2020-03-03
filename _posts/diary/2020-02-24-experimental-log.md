---
layout: post
title: "Experimental logs from 2/24/2020 - 3/1/2020"
author: Guanlin Li
tag: diary
---

- toc
{:toc}


### Feb. 24 2020

> 3:05pm, outside BIIIG sunshine!

Currently, I have settled the `env` on fire server.

```bash
# when the datasets are uploaded, do relink
ln -s test.en-fr.en.pth test.en.pth
ln -s test.en-fr.fr.pth test.fr.pth

ln -s valid.en-fr.en.pth valid.en.pth
ln -s valid.en-fr.fr.pth valid.fr.pth
```

Then when loading the `mass_enfr_1024.pth` checkpoint, I encountered the following error:

```bash
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.0.out_lin.weight not found.
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.0.out_lin.bias not found.
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.1.out_lin.weight not found.                                
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.1.out_lin.bias not found.
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.2.out_lin.weight not found.                            
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.2.out_lin.bias not found.
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.3.out_lin.weight not found.                                                        
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.3.out_lin.bias not found.                                                          
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.4.out_lin.weight not found.                                                        
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.4.out_lin.bias not found.                                                          
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.5.out_lin.weight not found.                                                        
WARNING - 02/24/20 15:14:20 - 0:00:14 - Parameter encoder_attn.5.out_lin.bias not found.
```

```bash
RuntimeError: Error(s) in loading state_dict for TransformerModel:
        Unexpected key(s) in state_dict: "encoder_attn.0.out_lin.0.weight", "encoder_attn.0.out_lin.0.bias", "encoder_attn.0.out_lin.1.weight", "e
ncoder_attn.0.out_lin.1.bias", "encoder_attn.1.out_lin.0.weight", "encoder_attn.1.out_lin.0.bias", "encoder_attn.1.out_lin.1.weight", "encoder_att
n.1.out_lin.1.bias", "encoder_attn.2.out_lin.0.weight", "encoder_attn.2.out_lin.0.bias", "encoder_attn.2.out_lin.1.weight", "encoder_attn.2.out_li
n.1.bias", "encoder_attn.3.out_lin.0.weight", "encoder_attn.3.out_lin.0.bias", "encoder_attn.3.out_lin.1.weight", "encoder_attn.3.out_lin.1.bias",
 "encoder_attn.4.out_lin.0.weight", "encoder_attn.4.out_lin.0.bias", "encoder_attn.4.out_lin.1.weight", "encoder_attn.4.out_lin.1.bias", "encoder_
attn.5.out_lin.0.weight", "encoder_attn.5.out_lin.0.bias", "encoder_attn.5.out_lin.1.weight", "encoder_attn.5.out_lin.1.bias".
```

***Q*** I should make clear whether MASS can as well initialize the `encoder_attn` of the decoder both in theory and implementation.

**Simple solution** $\Rightarrow$ just add `strict=False` in the decoder state loading part:

```python
decoder.load_state_dict(dec_reload, strict=False)
```

**Simple solution to also load src-attn weights**  $\Rightarrow$ just comment the following codes:

```python
				# in src/model/__init__.py
    			for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
```





### Feb. 25 2020

> 6:53am in the morning, hope it would be sunny as usual.

#### Vim-plug

```bash
# just upload the plug.vim file at my local machine
```

#### YouCompleteMe

```bash
# on FIRE, no cmake
brew install cmake
# then
python3 install.py --clang-completer  # using .linuxbrew/bin/python3 which is not the conda distribution that has trouble
```

#### How to save online BT data in `XLM`?

> This is used for understanding the initial boost of the translation quality:
>
> $\Rightarrow$ why does not learning completely fail given the initial BT data?
>
> $\Rightarrow$ How does AE loss help? Why can it be replaced by good initialization?

To get this work, I should check two code components at least:

1. data iterator parts through `load_data` function;

   - Maybe knowing how the `argument` are parsed and used are also important:

     ```bash
     --lgs 'en-fr'
     --ae_steps 'en,fr'
     --bt_steps 'en-fr-en,fr-en-fr'
     
     # after parsing the args, they become
     params.ae_steps = ['en', 'fr']
     params.bt_steps = [('en', 'fr', 'en'), ('fr', 'en', 'fr')]
     params.bt_src_langs = ['en', 'fr']
     ```

   > Note that [here](<https://epsilon-lee.github.io/blog/Experiment-Log/#6-2-2020>) is a my old notes on the data structure of `data` dict object when only doing Masked LM training.

2. `mt_step` function in `EncDecTrainer`

```python
# the `load_data` function
def load_data(params):
    """
    Load monolingual data.
    The returned dictionary contains:
    	- dico (dictionary)
    	- vocab (FloatTensor)
    	- train / valid / test (monolingual datasets)
    """
    
    # monolingual datasets
    load_mono_data(params, data)
    
    # parallel datasets
    load_para_data(params, data)
```

---

With UNMT or monolingual data training, the data name should look like:

```python
'%s.%s.pth' % (splt, lang)  # splt in ['train', 'valid', 'test'], lang in ['en', 'fr']

# ==>
# train.fr.pth, train.en.pth
```

---

Let's then look into the `bt_step` function.

```python
def bt_step(self, lang1, lang2, lang3, lambda_coeff):
    '''
    lang1 -> lang2 -> lang3: en -> fr -> en
    lambda_coeff: params.lambda_bt
    '''
    
    # generate source batch
    x1, len1 = self.get_batch('bt', lang1)
```

```python
def get_batch(self, iter_name, lang1, lang2=None, stream=False):
    
    # get iterator
    iterator = self.iterators.get((iter_name, lang1, lang2), None)
    
    # get batch
    try:
        x = next(iterator)
    except:
        # ...
```

To better know the iterator, we take look into the  `dataset.py` file.

```python
class StreamDataset(object):
    
```



### Feb. 26 2020

> It's very hot today, oh my god, about 30 degree centigrade.

#### Get Ruize's LCA code

How to zip folders in linux?

```bash
zip -r output_file.zip folder/ file

# or unzip a .zip file into a folder
unzip xxx.zip -d folder_name/
```

Determine public and private IP:

```bash
hostname -I
# =>
100.110.19.121 172.17.0.1
```

On my fire server:

```bash
# activate `xlm` conda env (ON FIRE)
CUDA_VISIBLE_DEVICES=1 python train.py /app/home/Workspace/fairseq_lca/iwslt14.tokenized.de-en --lr 0.05 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch transformer --save-dir checkpoints/transformer
```

On jizhi server:

```bash
# activate `th1.4`
CUDA_VISIBLE_DEVICES=1 python train.py /root/Workspace/fairseq_lca/iwslt14.tokenized.de-en --lr 0.05 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch transformer --save-dir checkpoints/transformer
```





### Feb. 27 2020

> My cat always jumps up on my keyboard so that I could not focus on working:)

#### Installing `fastText` on fire

It's really easy on fire, since I just use the command:

```bash
pip install fasttext
```

That's all! And the pybind11 is already satisfied, without `brew unlink gcc` trick.

#### Configure `JIZHI` for running `fairseq`

```bash
# upload .bashrc.jz file and mv .bashrc.jz /root/.bashrc

# create new environment for fairseq
conda create -n fairseq python=3.7

# when installing pytorch==1.4.0, apex cannot be compiled due to CUDA 10.1 but jizhi only have cuda 10.0
xxx # try to install pytorch==1.2.0, too slow
```

#### Update `gcc`

Currently on Jizhi server, `gcc` version is `4.8.5`, which is not compatible with `fairseq` when run `pip install -e .` in `fairseq/` folder. I totally followed [this](<https://www.jianshu.com/p/36f5d3524240>) blog's instructions. Till now, no error occurs, but still compiling...

So the result is: it works!

```bash
# new gcc
export PATH="/usr/local/gccc5.4/bin":$PATH
```



### Feb. 28 2020

> Amazon interview failed, but thanks to xintong and zhufeng, I accepted it and am happy now:)
>
> Just stand up where it failed!

#### Processing `NIST`

- Using `fastBPE`
- Using `sacreBLEU`

`PROBLEM`: I don't know why compiling fastBPE to binary with my linuxbrew `g++` will prompt the following error:

```bash
In file included from /data10/epsilonli/.linuxbrew/Cellar/gcc/5.5.0_7/include/c++/5.5.0/x86_64-unknown-linux-gnu/bits/c++config.h:489:0,
                 from /data10/epsilonli/.linuxbrew/Cellar/gcc/5.5.0_7/include/c++/5.5.0/utility:68,
                 from /data10/epsilonli/.linuxbrew/Cellar/gcc/5.5.0_7/include/c++/5.5.0/algorithm:60,
                 from fastBPE/fastBPE.hpp:3,                                                                                                      
                 from fastBPE/main.cc:1:                                 
/data10/epsilonli/.linuxbrew/Cellar/gcc/5.5.0_7/include/c++/5.5.0/x86_64-unknown-linux-gnu/bits/os_defines.h:39:22: fatal error: features.h: No su
ch file or directory                                                     
compilation terminated.
```

I have created the dataset of NIST-bpe32k, then I should move on to preprocess the raw bpe training data and binarize them.

```python
# preprocess.py
```



#### Bug when using `pytorch==1.2.0 `with `FusedAdam`

```bash
2020-02-28 16:12:29 | INFO | fairseq.optim.adam | using FusedAdam
./bash-script/nist.bpe32k.zh-en/train.run1.sh: line 15: 47710 Segmentation fault      (core dumped)
```

It is strange when using `pytorch==1.4.0`, `fairseq` will not use `fusedadam`, so that training will move on smoothly:

```bash
020-02-28 16:17:28 | INFO | fairseq.trainer | NOTE: your device may support faster training with --fp16
2020-02-28 16:17:37 | INFO | train | epoch 001:     19 / 3373 loss=14.704, nll_loss=14.68, ppl=26246.1, wps=33168.8, ups=2.2, wpb=15074.5, bsz=512, num_updates=20, lr=1.8375e-06, gnorm=7.015, clip=100, oom=0, train_wall=9, wall=24
```

In both `pytorch==1.2.0`:

```python
fused_adam_cuda = importlib.import_module("fused_adam_cuda")  # will throw exception
```

and then in next try:

```python
from apex.optimizers import FusedAdam as _FusedAdam  # since in `fairseq` env, I have installed apex, so this will executed successfully, but in `th1.4` env, I haven't installed apex, this will as well throw exception, result in nothing
```

***So the `core dumped` reason might be that `apex` is compiled with very low version of gcc, so might be incompatible with `pytorch` binaries.***

Since I am not using `apex`, I just delete it from `fairseq` env.



### Feb. 29 2020

> It Saturday today, but cloudy.

#### Write code for multi-ref test on NIST

1. Copy and Modify the `generate.py` file;
2. Add multi-reference evaluation script in it;

- **Method 1.** The easiest modification is to binarize each dev source files with `testprefs` given `--srcdict`.
- **Method 2.** Write a code that load the source and references files; word2id-ize the source, and create a iterator; then generate the target and restore order, then id2word-ize the predicted target, then call `sacrebleu` for computing BLEU values.



### Test `LCA` with TensorBoard

> This part is used for collaboration with my lab back in HIT.

There are several bugs that I struggle to run the code. Currently the biggest bug is not the speed of the LCA computation, but the stuck problem of LCA.





### Mar. 1 2020

> New month of 2020. Time flies!

#### Try to write multi-reference evaluation in `fairseq`

1. Make sure `sacrebleu` supports multi-reference BLEU calculation, if so, NICE, and know how;
2. Learn how to load bilingual files with single source file and multiple reference files;

3. Combine 1 and 2!

---

1. For Q1, I think `sacrebleu` supports multiple references, since its usage command line is:

```bash
cat output.detok.txt | sacrebleu REF1 [REF2 ...]
```

2. For Q2, I think it's time to write a new preprocessing code snippet for dealing with multiple references:

```bash
# The file format is:
dev.zh
dev0.en
dev1.en
dev2.en
dev3.en
```

Then, I add `args.multiref` to `options.py` in function `def add_preprocess_args(parser)`:

```python
group.add_argument("--multiref", default=1, type=int,
                  help="number of multiple reference files, if> 1, the file name format is valid0-3 for 4 references")
```



#### The `iterator` mechanism of `fairseq`

1. How is `epoch_itr` constructed? Or more specifically, how is `itr` in `def validate()` in `train.py` constructed?
2. What does `epoch_itr.next_epoch_itr()` do?

---

For Q1.

```python
# for the train loop iterator, it is constructed while try loading ckpt
extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

# and really constructed here in `checkpoint_utils.py`
if extra_state is not None and not args.reset_dataloader:
    # restore iterator from checkpoint
    itr_state = extra_state["train_iterator"]
    epoch_itr = trainer.get_train_iterator(
    	epoch=itr_state["epoch"],
        load_dataset=True,
        **passthrough_args
    )
    epoch_itr.load_state_dict(itr_state)
else:
    epoch_itr = trainer.get_train_iterator(
    	epoch=0,
        load_dataset=True,
        **passthrough_args
    )

# and the trainer.get_train_iterator() function is going to call
task.get_batch_iterator(
	dataset=self.task.dataset(self.args.train_subset),
    # ...
    epoch=epoch,
)
```

The validation iterator is constructed right in `validate()` function:

```python
itr = task.get_batch_iterator()

# in `fairseq_task.py`, the above function finally returns
epoch_iter = iterators.EpochBatchIterator(
	dataset=dataset,
    collate_fn=dataset.collater,
    batch_sampler=batch_sampler,
    # ...
    epoch=epoch
)

# before constructing `EpochBatchIterator`
with data_utils.numpy_seed(seed):
    indices = dataset.ordered_indices()  # sorted by length of each instance

if max_positions is not None:
    indices = data_utils.filter_by_size(
    	indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs)
    )  # filtering too long instances
    
# batch_sampler: a list of list of sample_id
[
    [6482, 5402, 4000],
    [3506, 5138]
]
```

The `CountingIterator` object is then constructed through `EpochBatchIterator.next_epoch_itr()` function which can be iterated through `next` operator.





### Mar. 3 2020

> The temperature starts to drop, take care scar!

#### Summary of last week's work

##### By item

1. Demystifying the Learning of UNMT

   a. <u>分析实验的设计</u>**没有**进一步进展（认为上周讨论的内容比较跳跃、实验假设是否可靠还未通过阅读文献验证）;

   b. `model.pretrain.5M`, `model.pretrain.fb`, `emb.pretrain.joint5M`和`model.pretrain.mass`的基础上，额外又跑了两组：`random`和`model.pretrain.mass-attninit`，实验结果基本出来汇总如下：

   | init. setting                  | En->Fr | Fr->En | avg. dev |
   | ------------------------------ | ------ | ------ | -------- |
   | `random`                       | 9.54   | 9.89   | 7.87     |
   | `emb.pretrain.joint5M`         | 27.94  | 26.21  | 23.85    |
   | `emb.pretrain.muse5M`          | -      | -      |          |
   | `model.pretrain.5M`            | 32.05  | 30.21  | 27.43    |
   | `model.pretrain.fb`            | 35.94  | 33.59  | 30.24    |
   | `model.pretrain.mass`          | 35.33  | 32.91  | 29.81    |
   | `model.pretrain.mass-attninit` | 35.12  | 32.98  | 29.83    |

   > $*$*待绘制Loss与performance随update和epoch的曲线.*

   - 另外预计还会加入一种用MUSE学cross-lingual embedding作为`emb.pretrain.muse5M`的初始化方法；
   - 基于上述三种init. settings，还会结合对应的论文中的ablation，进行补充ablation实验：s
     - emnlp best paper汇报说基于`emb.pretrain.joint5M`，去掉`denosing loss`，会导致模型性测试BLEU得分为0；另MASS汇报说去掉`bt loss`同样能训练处有效的模型，且汇报说他们的结果是去掉`denosing loss`后训练得到的 $\Rightarrow$ 分别去掉`bt loss`和`denosing loss`，在三种init. settings下测试（一共6组实验）；
       - 实验目的：充分认识learning protocol中各个loss的重要性；

   c. 完成了online搜集BT训练数据的code，可在上述有***明显差异***的**三种**setting下：random, emb init.与model.pretrain进行前2000*updates*的Co-Training数据收集；

2. Generalization Barriers

   a. 重新熟悉`fairseq`的最新repo，完成multi reference validation的code；

   b. 处理NIST数据，并完成base Transformer训练，nist02上BLEU49+，符合预期；

   c. 开始写risk estimation的code；

3. LCA

   a. 在`IWSLT14 DE-EN`测试速度，单块V100一个epoch约4min，同样batch size情况下，`lcaboard`的代码速度大约慢4~5倍（将在不同batch size做进一步速度测试）；另，`fairseq` Transformer base在该数据上性能为：`dev: 35.14, test: 34.35`，是很强的baseline，基本和目前改进训练策略的SOTA一致；

   b. 测试`lcaboard`，发现几个相关bug修改或汇报给润泽（iterator锁死的bug比较关键）；

   c. 规划LCA后续工作，将朱老师的LCA code适应于新的`fairseq` codebase中，准备测试后给两位师弟使用；

##### By day

**Wednesday**

1. Reading [Improving Generalization by Controlling Label-Noise Information in Neural Network Weights](<https://arxiv.org/abs/2002.07933>), ICML submissions; try to figure out how it relates generalization (label memorization/noise memorization) with conditional mutual information $\mathbb{I}(w:y \vert x)$.

2. Try to use a similar idea of the paper and apply a quantity of <u>label-noise (signal-noise) ratio</u> to measure the effectiveness of online Co-Training stage in UNMT; and this quantity during the ***process*** of online Co-Training shows how the model under different initializations start to teach themselves with <u>effective</u> learning supervised signals.

   - Most importantly, at convergence, this quantity can help rank the effectiveness of different initialization methods: since in my hypothesis, higher signal-noise ratio will finally results in higher performance.

   - > **Comment.**
     >
     > To some extend, this signal-noise ratio might be a very straightforward quantity, since if we are using the bilingual corpus to construct monolingual corpus, we can have golden bilingual corpus naturally for each $x^l \in \mathcal{D}^l$, and the training performance, say BLEU on training set which is similar to signal-noise ratio (if using BLEU-1 as the measure and implicitly measured by bilingual corpus) can directly reflect the generalization of the model, thus the test performance.

3. Thinking from theory of Co-Training;



**Thursday**

1. Configure server;
2. Write code to collect online BT data during unmt training;
3. Run unmt training under two more settings:
   - Using MASS checkpoint and adding source attention initialization $\Rightarrow$ similar performance;
   - Using random initialized model to do unmt training;



**Friday**

1. Preprocess NIST and start run `fairseq` (using the newest committed repo on github);
2. Start writing code for supporting multiple reference validation;



**Saturday**

1. Test `fairseq` on IWSLT14 DE-EN data, find that on single V100, speed is <4min each epoch, so the speed of current implementation is ***slow***;
2. Debug `LCA`, find several bugs and report them to `Runze`;
3. Check old code for `LCA`, ***adapt*** it to the newly version `fairseq` repo (used for collaboration with my lab);



**Monday**

1. Finish multiple reference validation code and rerun on NIST, initial BLEU on NIST02 is 49+, which reaches our expectation;
2. Start writing code for risk estimation with (using BLEU) and without reference (using model score, the score generated by `generate.py`);
3. One of the earliest pioneer in unmt upload a new paper on arXiv, which might be relevant to our investigation:
   - [Do all Roads Lead to Rome? Understanding the Role of Initialization in Iterative Back-Translation](<https://arxiv.org/pdf/2002.12867.pdf>). arXiv Mar. 2 2020, Mikel Artetxe et al.
     - This paper test under different initializations: supervised with small/large data, different warmup model architectures; unsupervised with SMT; do not change much the resulting model of continual iterative BT training;







