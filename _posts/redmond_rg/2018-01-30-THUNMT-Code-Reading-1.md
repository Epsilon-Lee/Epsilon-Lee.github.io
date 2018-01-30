---
layout: post
title: "THUNMT Code Reading Part 1: Execution of Functional Flow"
author: Guanlin Li
tag: redmond_rg
---

> [THUNMT](https://github.com/thumt/THUMT) is an open-source neural machine translation (nmt) code base or toolkit for fast prototype development. Since our team is doing research on nmt or more general structured prediction tasks. We have chosen THUNMT as our research prototype, so many thanks to [Yang Liu's team](http://nlp.csai.tsinghua.edu.cn/~ly/). 
>
> This post is about understanding the basic functional flow of the code, that is the execution flow from the program entrance. Since I am quite new to TensorFlow (which is THUNMT's underlying framework), most of the words below is more a learning process record than a deeper coding trick interpretation. However, hope it will help you better understand the code functionally and enjoy reading. More specifically, this post mainly focus on the `trainer.py` code.

[TOC]

### 1. `main` function in `thunmt/bin/trainer.py`

Almost all (supervised) deep learning training process is abstracted as 3 main components:

1. Preparation for training
2. Training
3. Validation and model selection

The THUNMT code is well-written and has a clear structure where all the preparation (except for dataset preprocessing which is done with code files in `thunmt/scripts`) for training is done within the `main` function in `thunmt/bin/trainer.py`. 

#### A. Preparation for training

> **Note that** this part mainly focus on code at the beginning of `train.py` file's `main()` function. 

This phase includes code before validation data loading, which is in **between** the `def main():` and the following code:

```python
# Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None
```

More specifically, this part contains the following main functionalities:

- **Training phase hyper-parameter configuration**
- **Computational graph construction**
- **Loss construction for multi-GPU**
- **Iteration counter construction**: `global_step = tf.train.get_or_create_global_step`
- **Optimizer construction**: using `tf.train.AdamOptimizer`
- **Train step construction**: get the `train_op` which contains the computation that is executed at each iteration/model update, based on the computation graph `tf.Graph().as_default()`. 

The preparation part deals with **training phase hyper-parameter configuration** which includes the model hyper-parameter configuration as well. There are three hyper-parameter source:

- `default_parameters()`: returns `tf.contrib.training.HParams` object, which includes several common training settings, such as initialization method, learning rate (decay), gradient clip threshold, maximum training steps etc. 
- Command line arguments through `argparse`: `args` returned by `parse_args()` function at the beginning of the `trainer.py` file. 
- Model-specific parameters: `model_cls.get_parameters()`. 

Note that, the model hyper-parameters are defined within the model class `model_cls` which is returned through `models.get_model(args.model)`, which follows an object-oriented design philosophy. 

Those parameters (you can see it as a dictionary) are then merged through the `merge_params()` function. If there is already checkpoint files saved under `params.output` directory, the `import_params()` function will override the parameters once. And then the `override_parameters(params, args)` will use `args` values to override the newly constructed or loaded `params` once again. The priority from low to high is `default->saved->command` which is in the comment of the code as follows:

```python
model_cls = models.get_model(args.model)
params = default_parameters()
# Import and override parameters
# Priorities (low -> high):
# default -> saved -> command
params = merge_parameters(params, model_cls.get_parameters())
params = import_params(args.output, args.model, params)
override_parameters(params, args)
```

After overriding the parameters, the program export the newly updated parameters to the disk once again. 

```python
# Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )
```



#### B. Training process

More accurately speaking, the training process is automatically managed by TensorFlow through its `tf.train.MonitoredTrainingSession()` class. So a deeper understanding of the controllable or customizable part of the actual training loop requires a better understanding of the functionality and source code of `MonitoredTrainingSession()` or the `Session()` class of TensorFlow (TF). I leave it to be a future post. Here, I discuss the hook mechanism which is enabled by the `MonitoredTrainingSession()` class. 

> **Note that**, actually, `tf.train.MonitoredTrainingSession()` is a function instead of a class defined in `monitored_session.py` file of the TF source code. And it returns a `MonitoredSession` class instance. 

At the end of the `main()` function of `train.py`, the code snippet is:

```python
# Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            while not sess.should_stop():
                # Bypass hook calls
                utils.session_run(sess, zero_op)
                for i in range(1, params.update_cycle):
                    utils.session_run(sess, collect_op)
                sess.run(train_op)
```

Here, we focus on the `sess.run(train_op)` line within the `while` loop. Apparently, the loop is not explicitly set to run some number of times, it will stop until the session detect a stop signal through a hook: `tf.train.StopAtStepHook(last_step=params.train_steps)`. So the training session will stop till the `global_step` reaches the number of `params.train_steps`. 

> **hook.** I am not familiar with hook as well. However, intuitively, I interpret hook as a callback function which will be called from the main loop when the main loop reaches or satisfies certain condition, like here, the `global_step` reaches `params_train_steps`. I think the design of hook mechanism is very elegant for high level abstraction of the variability of a main loop or normal routine such as here the training loop, but it abstract direct control away from us over the training loop, which is not the case in PyTorch where a `train_epoch` function is always defined manually. 

After getting an intuitive understanding of hooks, let us see what other kind of hooks does the code constructed. All the hooks constitute a python `list` object `train_hooks`:

```python
# Add hooks
        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "source": tf.shape(features["source"]),
                    "target": tf.shape(features["target"])
                },
                every_n_iter=1
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=params.output,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=tf.train.Saver(
                    max_to_keep=params.keep_checkpoint_max,
                    sharded=False
                )
            )
        ]
```

Here 4 hooks are constructed, namely `StopAtStepHook`, `NanTensorHook`, `LoggingTensorHook` and `CheckpointSaverHook`. Those hooks can access the computational graph during training, so it can detect whether the value of the `loss` node becomes `nan`; it can fetch the value of graph variable `global_step`, `loss`, `tf.shape(features["source"])` and `tf.shape(featuresp["target"])`. The printed log in the terminal will looks like. It is printed every `every_n_iter=1` iteration:

```shell
INFO:tensorflow:step = 1, loss = 9.42072, source = [128  20], target = [128  20]
INFO:tensorflow:Saving checkpoints for 1 into train/model.ckpt.
INFO:tensorflow:step = 2, loss = 9.0781, source = [128   8], target = [128   8] (5.854 sec)
INFO:tensorflow:step = 3, loss = 9.45908, source = [128  24], target = [128  24] (0.801 sec)
INFO:tensorflow:step = 4, loss = 9.32748, source = [128  14], target = [128  14] (0.568 sec)
INFO:tensorflow:step = 5, loss = 9.53357, source = [128  40], target = [128  40] (1.258 sec)
INFO:tensorflow:step = 6, loss = 9.18831, source = [128  10], target = [128  10] (0.454 sec)
INFO:tensorflow:step = 7, loss = 9.24082, source = [128  12], target = [128  12] (0.503 sec)
INFO:tensorflow:step = 8, loss = 9.34216, source = [128  16], target = [128  16] (0.609 sec)
INFO:tensorflow:step = 9, loss = 9.39888, source = [128  20], target = [128  20] (0.707 sec)
INFO:tensorflow:step = 10, loss = 9.43514, source = [128  28], target = [128  28] (0.934 sec)
```

#### C. Validation and model selection

> Actually, this part is closely related to the training process, since thunmt package provide the functionality for model selection **during training** through the `EvaluationHook`. 

The checkpoint hook is used for saving checkpoint. This hook is necessary when we perform **no** evaluation during training, that is we set `eval_input_fn` as `None`: 

```python
# Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None
```

If `eval_input_fn` is not `None`, the `EvaluationHook` is constructed through the following code. 

```python
        if eval_input_fn is not None:
            train_hooks.append(
                hooks.EvaluationHook(
                    lambda f: search.create_inference_graph(
                        model.get_evaluation_func(), f, params
                    ),
                    lambda: eval_input_fn(eval_inputs, params),
                    lambda x: decode_target_ids(x, params),
                    params.output,
                    config,
                    params.keep_top_checkpoint_max,
                    eval_secs=params.eval_secs,
                    eval_steps=params.eval_steps
                )
            )
```

`EvaluationHook` needs three functions to initialize it self:

- `search.create_inference_graph(model.get_evaluation_func(), f, params)`: a search module performs beam search in decoding. 
- `eval_input_fn(eval_inputs, params)`: a Tensor iterator over the evaluation data. 
- `decode_target_ids(x, params)`: an index to symbol transformation function. Thus we can get strings out of ids of the prediction produced by the inference graph. 

Note that the `EvaluationHook` is defined by the author in `thunmt/utils/hooks.py`. This customized hook follows the hook definition specification written [here](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook). 

```python
class EvaluationHook(tf.train.SessionRunHook):
    """ Validate and save checkpoints every N steps or seconds.
        This hook only saves checkpoint according to a specific metric.
    """

    def __init__(self, eval_fn, eval_input_fn, eval_decode_fn, base_dir,
                 session_config, max_to_keep=5, eval_secs=None,
                 eval_steps=None, metric="BLEU"):
        """ Initializes a `EvaluationHook`.
        :param eval_fn: A function with signature (feature)
        :param eval_input_fn: A function with signature ()
        :param eval_decode_fn: A function with signature (inputs)
        :param base_dir: A string. Base directory for the checkpoint files.
        :param session_config: An instance of tf.ConfigProto
        :param max_to_keep: An integer. The maximum of checkpoints to save
        :param eval_secs: An integer, eval every N secs.
        :param eval_steps: An integer, eval every N steps.
        :param checkpoint_basename: `str`, base name for the checkpoint files.
        :raises ValueError: One of `save_steps` or `save_secs` should be set.
        :raises ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EvaluationHook.")

        if metric != "BLEU":
            raise ValueError("Currently, EvaluationHook only support BLEU")

        self._base_dir = base_dir.rstrip("/")
        self._session_config = session_config
        self._save_path = os.path.join(base_dir, "eval")
        self._record_name = os.path.join(self._save_path, "record")
        self._log_name = os.path.join(self._save_path, "log")
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._eval_decode_fn = eval_decode_fn
        self._max_to_keep = max_to_keep
        self._metric = metric
        self._global_step = None
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=eval_secs or None, every_steps=eval_steps or None
        )

    def begin(self):
        # ... details omitted

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step) # get the int value

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Save model
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._eval_decode_fn,
                                  self._base_dir,
                                  self._session_config)
                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)
				
                if added is not None:
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_path, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_path, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_path
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))

    def end(self, session):
        # similar to the above after_run() method
```

During training, there are `params.keep_checkpoint_max` checkpoints saved to the disk, along with their BLEU scores. And every time a new evaluation result is calculated, the evaluation hook will check whether to save the current model and delete checkpoints which fall out of the top `params.keep_checkpoint_max`. 





























