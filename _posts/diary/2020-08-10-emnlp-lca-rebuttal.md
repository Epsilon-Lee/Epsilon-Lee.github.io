#### Response to Reviewer 1

"I feel that other claims made in this paper are more due to the LCA method than anything else"

Yes, when we first apply the LCA method for probing the learning dynamics of NMT model, our goal is to link certain learning patterns found by LCA to specific property of the NMT model architecture or linguistic characteristics of the translation task, though we are aware of our analyses might be only a subset of the analyses in the original LCA paper. The results turn out to be that specific conclusions can be drawn from specific model arch. as well, e.g. like you mentioned the sandwich effect of the dense weights (which is not found by the original paper). And the convergence speed (fast-to-slow) of the layer from bottom-to-top is also rediscovered in our experiments (Figure 4). More than some interesting findings, we think one non-ignorable contribution of our paper is that the approximation of the whole train/set LCA via mini-batch is feasible and lead to low-variance LCA value estimation, which means we might use such approximation in other tasks involving large models and datasets.



"Poor citations"

Due to limited space, we are not able to include a thorough introduction to all model-intrinsic interpretability works. Since our works is mostly motivated from current poor understanding of the learning process of NMT model instead of interpreting a well-trained model's decision making, we have not include those works that are trying to better determine and visualize alignment-like input-prediction relationship, but we will definitely include them in extended version to put our work in better position.



"Writing style not great"

We will definitely improve our writing according to your advices, especially focusing on using more formal expressions.



"thought about normalizing LCA values by the number of updates"

Not really, but we have realized the LCA value for the embeddings weights is high correlated with update frequency or more precisely word frequency (Figure 5). This is actually denoted as frequency bias in [1]. We haven't include one experiment in our paper (due to space limitation) on randomizing and fixing the embedding matrix from beginning of training. We hypothesize that small LCA value might mean less importance the value of the learned weights, so the weight could be randomized without much loss in final performance: on IWSLT corpus, the average BLEU drop is less than 1 BLEU point indicating that the learning of specific embedding weights is not so important, the model only need to distinguish one word from another, and this can be achieved even with randomized embeddings.



[1]. FRAGE: Frequency-Agnostic Word Representation, NeurIPS 2018.



"sparse weights, dense weights"

Yes, sparse weights refer to the word embedding matrix for the lookup operation. And dense weights refer to everything else. As the name suggests, sparse means it is updated sparsely instead of being updated every iteration. We will make the usage of those terms more clear in next revised version.



#### Response to Reviewer 2

"It is unclear what to conclude from these findings"

We think there are a few interesting conclusions or future works that can be inspired by our work:

1. We have obtained a better understanding on the convergence speed of each layer's weights as well as its contribution to loss degradation, this might hint on better optimization for those lower-layers.
2. We might further investigate the weights that related to certain attention head and link head's LCA to the head sparsity findings in [1] and [2].
3. We also find that the ranking of loss contribution of each group of weights is determined early in during training, this finding could be used to motivate online structural pruning method instead of current offline standard.

[1] Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned, EMNLP 2019.

[2] Are Sixteen Heads Really Better than One? NeurIPS 2019.





#### Response to Reviewer 3

"poor evaluation"

15 is a not a critical hyper-parameters since it does not change the ranking of the LCA values of different groups of weights which guarantees reasonable conclusions. Here we use 15 to get reasonable number of sampling points for reflecting the LCA trends over the whole training process.

We haven't tried other kind of groupings, since grouping by module is the most natural way of investigating each module's loss contribution as a whole. But we do think grouping by different attention heads is also of great interest due to some recent findings that most head can be pruned, which we would like to try in our future works.