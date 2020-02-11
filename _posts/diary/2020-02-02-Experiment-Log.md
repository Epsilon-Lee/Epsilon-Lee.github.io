---
layout: post
title: "Experiment Log from 2/2/2020 - 2/8/2020"
author: Guanlin Li
tag: diary
---

* [Configure `jizhi` server](#configure-`jizhi`-server)
* [Run XLM experiment](#run-xlm-experiment)
* [6-2-2020](#6-2-2020)
* [Answer specific Qs](#answer-specific-qs)
* [Train word embeddings](#train-word-embeddings)
  * [Cost of unlink gcc](#cost-of-unlink-gcc)
* [8-2-2020](#8-2-2020)

#### Configure `jizhi` server

```bash
# install miniconda
scp Software/Miniconda3-4.7.12.1-Linux-x86_64.sh $jz_server:/root/

# upload .bashrc.jz
scp .bashrc.jz $jz_server:/root/

# replace original .bashrc
mv .bashrc.jz .bashrc

# install liuxbrew
git clone https://github.com/Homebrew/brew ~/.linuxbrew/Homebrew
mkdir ~/.linuxbrew/bin
ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin
eval $(~/.linuxbrew/bin/brew shellenv)

brew  # installing brew
brew install tmux
brew instal vim

# upload .vimrc and :PlugInstall
cd ~/.vim/YouCompleteMe
export PATH="/usr/local/bin/python3":$PATH  # make sure to use system's python
python3 install.py --clang-completer
```

`0-9jizhi` plantform password: `zyrRjan@EKBwx1

##### Conda `python` env

```bash
conda create -n xlm python=3.7
conda activate xlm  # ca xlm

# install pytorch=1.0.1
pip install torch==1.0.1
pip install numpy
pip install ipdb

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

When installing `apex`, there is an error emerged:

```:Lbash
?>RuntimeError: Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries.  Pytorch binaries were compiled with Cuda 9.0.176.
```

So I instead try to install under `base` env and use:

```bash
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

# speed really sucks, use Tsinghua's conda resource via a .condarc file in home/
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

but install under `xlm` encounter following problem:

```bash
conda install --name xlm pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

==> WARNING: A newer version of conda exists. <==
  current version: 4.7.12
  latest version: 4.8.2

Please update conda by running

    $ conda update -n base -c defaults conda
```

So maybe I should update conda first. ***After executing the above commands for xlm, compiling of apex finally succeeded***.

---

#### Run XLM Experiment

```bash
export CUDA_VISIBLE_DEVICES=0,1; export RANK=0; export WORLD_SIZE=2; export NGPU=2;
export MASTER_ADDR=9.91.7.147;
export MASTER_PORT=8087;
./bash-script/pretrain.mlm.en-fr.sh  # this will take several minutes, don't know why
```

The above problem can be easily solved by carefully reading the `README.md` of the `XLM` repo in the [FAQ section](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus).

```bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

---

### 6-2-2020

Some components of the code `xlm` in terms of <u>cross-lingual pretraining</u>, i.e. *masked language modelling training*, on `en-fr` corpus:

- The basic functional flow;
- `Initialization`
- `load_data(params)`
- `TransformerModel`: `build_model(params, data['dico'])`
- `Trainer`

---

#### The basic functional flow

```python
# Into train.py __main__

check_data_params(params)
check_model_params(params)

# Into train.py main() function

# Initialize the multi-GPU / multi-mode training
init_distributed_mode(params)  # there are several modes for training, e.g. using SLURM or PyTorch's original torch.distributed

# Initialize the experiment
logger = initialize_exp(params)  # dump parameter's to the params.exp_name folder's params.exp_id folder, where store the multiple runs of the same experiment params.exp_name

# Initialize SLURM signal handler for time limit  / pre-emption
init_signal_handler()  # it seems like SLURM boilercode

# ----------

# Load data! Load data! Load data!
data = load_data(params)  # load monolingual data and parallel data with a dict contains 'dico', 'train/valid/test'

# Build model! Build model! Build model!
if params.encoder_only:
    # for pretraining cross-lingual model with causal LM or BERT like LM objective, we only have a single model for different languages with a shared parameterization as SAN arch.
    model = build_model(params, data['dico'])
else:
    # for seq2seq tasks like neural machine translation, we additionally have a SAN arch as a decoder with encoder attention besides
    encoder, decoder = build_model(params, data['dico'])

# Build Trainer! Build Trainer! Build Trainer!
if params.encoder_only:
    trainer = SingleTrainer(model, data, params)
    evaluator = SingleEvaluator(model, data, params)
else:
    trainer = EncDecTrainer(model, data, params)
    evaluator = EncDecEvaluator(model, data, params)

# If the user want to load the model for only evaluation, the user can set params.eval_only, which is a `type=bool_flag` to True
if params.eval_only:
    scores = evaluator.run_all_evals(trainer)
    for k, v in scores.items():
        logger.info("%s -> %.6f" % (k, v))
    logger.info("__log__: %s" % json.dump(scores))
    exit()

# Set the probability of sampling specific languages / language pairs during training
set_sampling_probs(data, params)

# Then the training begins
for _ in range(params.max_epoch):  # run at most that many epochs if stopping criteria is not matched
    
    trainer.n_sentences = 0
    
    # for cross-lingual pre-training type of works, the code usually assign a epoch_size to indicate how many sentences are seen and learned from by the model to trigger the end of an epoch
    while trainer.n_sentences < trainer.epoch_size:  
        # continue training
        
        # ...
        for lang1, lang2 in shuf_order(params.mlm_steps, params):
            trainer.mlm_step(lang1, lang2, params.lambda_mlm)
        # ...
        
        trainer.iter()  # print log info:
```

```bash
INFO - 02/06/20 09:30:22 - 0:00:16 -       5 -   56.87 sent/s -  2164.25 words/s - MLM-en:  9.6898 || MLM-fr:  8.8403 -  - model LR: 1.0000e-04
```

```python
    # After each epoch, run evaluation and save model
    scores = evaluator.run_all_evals(trainer)
    
    # print / JSON log
    # ...
    
    # end of epoch
    ## Compare to history scores in trainer.metrics and save current if it has got the best scores, else do nothing
    trainer.save_best_model(scores)
    # Save every params.save_periodic epoch(s)
    trainer.save_periodic()
    # Stop if the stopping criterion has not improved after certain numbers of epochs, also save the model after every epoch
    trainer.end_epoch(scores)
```

---

#### Data structure of `data` from `load_data(params)`

```python
data.keys()
# => ['mono', 'mono_stream', 'dico', 'para']
data['mono'].keys()
# => ['en', 'fr']
data['mono']['en']
# => {} null dict, as well as data['mono']['fr']; data['mono'] stores nothing

# ----------

data['mono_stream'].keys()
# => ['en', 'fr']
data['mono_stream']['en'].keys()
# => ['train', 'valid', 'test']
type(data['mono_stream']['en']['train'])
# => <class 'src.data.dataset.StreamDataset'>

dir(data['mono_stream']['en']['train'])
# =>
```

```bash
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'bptt', 'data', 'eos', 'get_iterator', 'lengths', 'n_batches', 'n_sentences', 'n_tokens', 'select_data']
```

```python
data['mono_stream']['en']['train'].data.shape
# => (4184065, 32)

data['param']  # => {}, since in current pretraining, no parallel data is used
type(data['dico'])
# => <src.data.dictionary.Dictionary object at 0x7fca46876090>
dir(data['dico'])
# =>
```

```bash
['__class__', '__contains__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'bos_index', 'check_valid', 'counts', 'eos_index', 'id2word', 'index', 'index_data', 'max_vocab', 'min_count', 'pad_index', 'read_vocab', 'unk_index', 'word2id']
```

---

#### `build_model()`

<u>Create the object</u> of `TransformerModel` according to `params.encoder_only`, and then <u>load checkpoints</u> (*pretrained model weights*).

```python
# in src/model/__init__.py build_model(params, dico)
if params.encoder_only:
    model = TransformerModel(params, dico, is_encoder=True, with_output=True)
    
    # reload pretrained word embeddings
    if params.reload_emb != '':
        # ...
        
    # reload a pretrained model (Note: it seems to cover the reload emb.)
    if params.reload_model != '':
        logger.info("Reloading model from %s ..." % params.reload_model)
     	# ...
else:
	encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)
    decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)
    
    # reload pretrained word word embeddings
    if params.reload_emb != '':
        # ...
        
    # reload a pretrained model
    if params.reload_model != '':
        enc_path, dec_path = params.reload_model.split(',')
        assert not (enc_path == '' and dec_path == '')  # at least one is not null
        
        # Here note that the encoder and decoder are saved separately within the pretraining stage
        
        if enc_path != '':
            # ...
        if dec_path != '':
            # ...
```

---

### Answer specific Qs

- **Q1**. How to reload from checkpoint? How to restore from `ctrl-c`interruption?

In the `get_parser()` function in `train.py`, there is a code block that dedicates to loading weights from certain path as follows:

```python
parser.add_argument("--reload_emb", type=str, default="",
                   help="Reload pretrained word embeddings")
parser.add_argument("--reload_model", type=str, default="",
                   help="Reload a pretrained model")
parser.add_argument("--reload_checkpoint", type=str, default="",
                   help="Reload a checkpoint")
```

So according to the `--reload_checkpoint` argument, I think it would be easy to restore training once this is assigned, for example:

```bash
export NGPU=4
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
	--exp_name enfr_mlm \
	--exp_id test \
	--dump_path ./dumped/ \
	--data_path ./data/processed/en-fr/
	--lgs "en-fr" \
	--clm_steps "" \
	--mlm_steps "en,fr"\
	--emb_dim 1024 \
	--n_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--attention_dropout 0.1 \
	--gelu_activation true \
	--batch_size 32 \
	--bptt 256 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 200000 \
	--validation_metrics _valid_mlm_ppl \
	--stopping_criterion _valid_mlm_ppl,10 \
	--reload_checkpoint './dumped/enfr_mlm/test/checkpoint.pth'
```

- **Q3**. How to reload the embedding matrix into the enc-dec model for UNMT fine-tuning?

<u>Reloading (the pretrained model weights)</u> is written in the `build_model()` function in the `src/model/__init__.py` file. Current code *enables*:

**a.** Reloading from a `fastTEXT` like pre-trained embeddings; 

**b.** Reloading from a pretrained encoder and/or decoder with the same built architectures.

Let's see how the code enables these utilities.

**Reloading from pretrained embeddings**

Two functions perfectly finish the job:

- `load_embeddings(params.reload_emb, params)`
- `set_pretrain_emb(model, dico, word2id, embeddings)`

```python
def load_embeddings(path, params):	# -> word2id, embeddings
    '''...'''
    if path.endswith('.bin'):  # binary file for embeddings
        return load_bin_embeddings(path, params)
    else:  # else raw text file for embeddings
        return load_txt_embeddings(path, params)
```

```python
def load_bin_embeddings(path, params):
    model = load_fasttext_model(path)
    assert model.get_dimension() == params.emb_dim
    words = model.get_labels()  # the type of `words` is crucial
    
    # compute new vocabulary / embeddings
    embeddings = np.concatenate(
        [model.get_word_vector(w)[None] for w in words],
        0
    )  # [V, emb_dim]
    embeddings = torch.from_numpy(embeddings).float()
    word2id = {w: i for i, w in enumerate(words)}
    
    assert embeddings.size() == (len(word2id), params.emb_dim)
    return word2id, embeddings
```

```python
def set_pretrain_emb(model, dico, word2id, embeddings):
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            # initialize both the input and output embedding
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    # ...
```

> To note that, here the code use another external library `fasttext`.
>
> ```python
> def load_fasttext_model(path):
>     try:
>         import fastText
>     except ImportError:
>         raise Exception("Unable to import fastText. Please install fastText for Python: https://github.com/facebookresearch/fastText")
>     
>     return fastText.load_model(path)
> ```

For finetuning the encoder-decoder architecture, the `set_pretrain_emb(xxx, dico, word2id, embeddings)` function should be called for <u>**2 times**</u> with `xxx=encoder or decoder` respectively.

**Reloading from pretrained model to `encoder`**

```python
reloaded = torch.load(
	params.reload_model,
    map_location=lambda storage, loc: storage.cuda(params.local_rank)['model']
)
```



**Reloading from pretrained model to `enc-dec`**



---

### Train word embeddings

According to the emnlp18 best paper [Phrase-based & Neural UNMT]:

> *First, instead of considering words, we consider byte-pair encoding, which have two major advantages: they reduce the vocabulary size and they eliminate the presence of unknown words in the output translation. Second, instead of learning an explicit mapping between BPEs in the source and target languages, <u>we define BPE tokens by jointly processing both monolingual corpora</u>.*
>
> *"If languages are related, they will naturally share a good fraction of BPE tokens, which eliminates the need to infer a bilingual dictionary."*
>
> *"In practice, we i) join the monolingual corpora, ii) apply BPE tokenization on the resulting corpus, and iii) learn token embeddings (fastText) on the same corpus, which are then used to initialize the lookup tables in the encoder and decoder."*

Currently, we have in `data/processed/en-fr` folder: the `vocab.en-fr`, which is the joint vocabulary of the two languages; the `codes` which is the joint BPE code book for apply BPE tokenization to new raw text files; <u>the `train.en` and `train.fr` the BPE-tokenized monolingual corpora, which can be concatenated for training `fastText` under BPE tokenizastion.</u>

---

> `Some trouble installing fastText (python)` => `pybind11` is needed

Try to using `brew install pybind11`... ***And it works |^_^|***

However, when running the following script:

```python
from fasttext import train_unsupervised
```

the error occurred:

```bash
ImportError: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by /data10/epsilonli/.linuxbrew/lib/libstdc++.so.6)
```

But I found a trick to solve this problem, which is weirdest.

```bash
# first I try to reinstall glibc in brew, since I though brew does not have the glibc lib, so python uses system's
brew install glibc
# =>
Warning: glibc 2.23 is already installed, it's just not linked

# so I try to link to glibc
brew link glibc
# =>
Linking /root/.linuxbrew/Cellar/glibc/2.23...
Error: Could not symlink lib/libgcc_s.so.1
Target /root/.linuxbrew/lib/libgcc_s.so.1
is a symlink belonging to gcc. You can unlink it:
  brew unlink gcc

To force the link and overwrite all conflicting files:
  brew link --overwrite glibc

To list all files that would be deleted:
  brew link --overwrite --dry-run glibc

# so I unlink gcc
brew unlink gcc
# =>
Unlinking /root/.linuxbrew/Cellar/gcc/5.5.0_7... 106 symlinks removed
# which means I am using system's gcc afterwards

which gcc
# => /usr/bin/gcc

# And I try to import fasttext in fasttext, and it works
# =>
\*_^/ ipython
Python 3.7.6 (default, Jan  8 2020, 19:59:22)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.12.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import fasttext

In [2]:
```

This means I am not possible to link the glibc library in my linuxbrew installation, due to link to linuxbrew's `gcc`, which I don't know why. And unlink brew's `gcc` save my life for using `fastText`.

---

#### Cost of unlink gcc

I find a cost of unlink `gcc` when I try to get `fastText` (python bind) to work properly. That is, the YouCompleteMe's `find definition` functionality stops working. But when I relink the brew's `gcc`, it gets back to life. ***I really don't know WHY?***

---

### 8-2-2020

1. Carefully check how to load a model;
2. Run the finetuning experiment with the pretrained model;

---

**Reloading from GPU (single/multi-GPUs)**

The first problem is reloading to a single GPU with multi-GPU trained model. See [this](<https://github.com/facebookresearch/XLM/issues/51>) issue for more information.

---

**Finetuning unmt**

There are two issues in the `XLM` repo which points to some possible stuck situation.

- [ISSUE A: The UNMT Training On Multi-GPU Report Errors](<https://github.com/facebookresearch/XLM/issues/95>)
- [ISSUE B: Parallel training of unsupervised mt model fails](<https://github.com/facebookresearch/XLM/issues/211>)
- [ISSUE C: Error when using multi-GPU for training MT only](<https://github.com/facebookresearch/XLM/issues/41>)

