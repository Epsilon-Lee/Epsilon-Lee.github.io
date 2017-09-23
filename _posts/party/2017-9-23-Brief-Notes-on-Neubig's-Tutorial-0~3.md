---
layout: post
title: "Brief Notes on Neubig's Tutorial 0~3"
author: Guanlin Li
tag: party
---

### 1. Tutorial 0 - Setting up your environment and basic python

- The goal of this course: 
  - During tutorial: learn something new: new concepts, new models, new algorithms etc. 
  - And our adapted goal is: run the program during the tutorial, and let all the members watch and see the results. 
  - Another encourage from Neubig is to work in pairs or small groups, this enables discussion and cooperation which will make every one within the group to learn 2 or 3 times efficiently and more. 
- Programming language is not restricted to Python. However, I would really recommend Python as your quick working language or prototype language with which you can quickly develop some raw idea upon. Another reason is that python is so hot recently because of most of the deep learning framework supported python. 

#### Setting up environment

It is very important for one to get familiar with his/her most suitable development environment. During this section Neubig gives some specific advice which I would like to add my owns. 

- **Be able to access to Google!** This is very important since every time you want to look up something, maybe a command line or just some bug information prompted from you runtime python program. Just type/copy it in the Google search textarea. This will open a window of independence for yourself. 


- **Be familiar with linux terminal environment.** In the future, most of your jobs will be working with Linux, typing `ls /usr/local/cuda` , `cd ~/Workspace/Experiment` `ipython`, `mkdir`, `mv` etc. 

  - **Login to a Linux machine.** If you have a server id, like me, I have an account on 117 server which is on the LAN of our lab, you can log in on that machine via `ssh glli@10.10.255.117`(this is my log in command). Then you will be required for your password. But I would recommend using a linux client named [MobaXterm](http://mobaxterm.mobatek.net/) which is an integrated client app that can hold a file management GUI and a terminal window, like the following. Since we call a connection between your local machine and the linux server as a session, you can manage your sessions through within the same Tab view, like tab management in Chrome and other browser. 

	![]({{ site.baseurl }}/public/img/party/mobaXterm.png)


  - **Create a session.** The way to create a session is very simple, just click `Session` button in the tool bar, the first button along that row. And choose `ssh` connection and then type your `Remote host` IP and your username in `Specify username`. Then you can click OK to go on opening a terminal and input your password to log in. *Note that: this session will saved and you can find it at the session tab at the left edge of your window with a star logo! You won't miss it! Every time you want to create a new terminal, just double click your session. You can rename the session as I do in the following.* 

	![]({{ site.baseurl }}/public/img/party/session_ssh.png)


	![]({{ site.baseurl }}/public/img/party/session_rename.png)

  - **Use your favorite text editor.** I recommend [Sublime Text](https://www.sublimetext.com/) if you are not familiar with vim/gvim etc. Then your coding workflow will be like this: a). Open a source file through the `sftp` tab of MobaXterm using sublime; 2). Edit it, and save it using `ctrl+s`; 3). Debug it in the terminal, and go back to a). until you finish your work. 

- **Use `git` to clone the repo.** Firstly, if you don't have a [github](https://github.com/) account please sign up one! (This does not bother you use `git`, a local app on your local machine or server, sign up one just because github is awesome.) In your local machine or server, `cygwin` or `terminal`, type `git clone https://github.com/Epsilon-Lee/nlptutorial.git` to get a local repo of the tutorial. If you are a fresh man on using git, try to learn more about it. I started this summer from [here](https://www.codecademy.com/learn/learn-git) at Code academy. 

#### Python basics

- **Python data types.** `str`, `int` and `float` etc. those are basic data types. You can use `print("%s %f %d" % ('nlp', .3, 1))` where `%s` to hold string, `%f` to hold float and `%d` to hold integer. 
- **Python condition statement.** `if ... elif ... else ...`. 
- **Python advanced built-in data structure.** `list` and `dict`. Know how to *create*, *access*, *assign* value to, *delete* value from, *traverse*, *sort* these two data structures. Try to use google to find every operations listed before. 
- **Python `collection` module.** There are many great and useful sub-modules in the `collection` module, you can refer to this [Chinese post](http://www.zlovezl.cn/articles/collections-in-python/) for more knowledge. 
- **Python read text file, split line, count vocabulary.** It is very common for researchers in NLP to create vocabulary from raw text files when working on a new task or project, so basic text file manipulation with python is essential for everyone. 
- **Unit test.** Since I am not a software engineer and have not contribute to something sophisticated, I just don't know the importance of doing unit test. But after I have some experience with a little bigger programs in Machine Learning, for example, a sequence-to-sequence attentional model for neural machine translation. It suddenly become necessary for me to modularize my project and test each modules with input-output guarantee! Otherwise, if I just run them as a whole, it would be hard for me to debug since I cannot remember every detail of the whole of my code. 

### 2. Tutorial 1 - Language model basics

#### Why language models?

This is a small question in terms of Neubig's slides content. However, he is just try to use some example that could make sense to beginners for introducing the language models based on its utility aspect, that is, language models could score the output of a speech recognition system help the system to response to its user the most natural recognized result. Of course, after making a little bit sense of the practical aspect of a language model, it is more valuable to know that language model is more than that! 

To answer what is a language model philosophically, we should understand what is a model in a probabilistic point of view. **Probabilistic model** or **model** is a probabilistic distribution, denoted as $$ \mathbb{P} (\cdot) $$, over certain sample space that we care most, so the distribution could be used to summarize statistical regularities and properties of the elements in the sample space, intuitively we call those elements as a kind of **phenomenon** from nature and human society. The usage of a model is to **open a statistical toolkit** for human to describe phenomenon (you can call phenomenon **data**), to understand some statistical or even causal aspect of phenomenon, or even to predict the occurrence of certain phenomenon of interest. 

Language model is used for human to summarize statistical/descriptive properties in language data! It is a probabilistic distribution over natural language, i.e. words, sentences, paragraphs etc. 

Specifically speaking, we can use language model to compute probability of a **word sequence** $$ w_1, \dots, w_n $$ if the minimum element of a language is word (this is not always true, since in Chinese or Japanese we can have characters which is subpart of a word.) We can get the symbolic representation as: 

$$ P(w_1, \dots, w_n) $$

There are many ways to parameterize the above probabilistic formula. 

> **Concept of parameterization.** Parameterization is a very important and practical concepts in probabilistic modeling methods. The meaning of parameterization is: a kind of representation of the computational aspect of the model. More specifically that is given an element in the sample space, e.g. here we get a word sequence $$ w_1, \dots, w_n $$, the computational process for calculating its probability. 
>
> Another important aspect of parameterization is that this concept is usually used for **parametric models** which is a probabilistic model that has some **parameters** whose values are determined/learned from data, i.e. fitting the model to the data or training the model, aka. machine learning! Intuitively speaking, if model is a family of a certain descriptive method for the data, a specific parameters setting decide a certain model in that family which has **knowledge** about its learning resource -- the data. So the knowledge from data is "memorized" in the parameters! 

One way to parameterize $$ P(w_1, \dots, w_n) $$ is to use the so-called bi-gram language model, which is depend on the 1st order Markov assumption as following. Since we can use probabilistic chain rule to decompose $$ P(w_1, \dots, w_n) $$ as: 



$$ P(w_1, \dots, w_n) = P(w_1) P(w_2 \vert w_1) P(w_3 \vert w_1, w_2) \dots P(w_n \vert w_1 \dots w_{n-1})$$



Then according to the 1st order Markov assumption, we get that: $$ P(w_i \vert w_{1:i-1}) = P(w_i \vert w_{i-1}) $$. 

> **k-th order Markov Assumption.** The Markov assumption means that the conditional probability of the current state $$ s_t $$ is independent with the history states if given its k previous states $$ s_{i-k}, \dots, s_{i-1} $$. 

So we can get: 



$$ P(w_1, \dots, w_n) = P(w_1 \vert \cdot) P(w_2 \vert w_1) P(w_3 \vert w_2) \dots P(w_n \vert w_{n-1}) $$



On the right hand side (RHS) of the above formula, it is very regular that each probability term has two specific symbols $$ P( symbol_1 \vert symbol_2) $$ with the conditional mark $$ \vert $$ in between. *Note that, here the first term of the RHS used to $$ P(w_1) $$, but now we want to every term to be of the same form, we re-denote it as $$ P(w_1 \vert \cdot) $$, where the $$ \cdot $$ is a null symbol, or you can think of it as a beginning symbol denoting a sequence is beginning!*

The 2-gram or bi-gram language model is parameterized by a group of conditional probabilistic terms with the above mentioned form. The parameters of this model is each conditional terms $$ P(\cdot \vert \cdot) $$ which has its value between $$ (0,1) $$. If we have a vocabulary of 10000, we will have $$ 10000^2 $$ number of such terms (easy combinatorial math! Right?) as our model's parameters. So if we have an accurate **estimation** of these parameters. We can use them and the decomposed formula to compute the probability of a sentence! For example, a sentence "I love my parents !" We can compute its probability according to our model as following: 



$$ P("\text{I love my parents !}") = P("\text{<s> I love my parents ! </s>}") = P("\text{<s>}") P("\text{I}" \vert "\text{<s>}") P("\text{love}" \vert "\text{I}") \\ P("\text{my}" \vert "\text{love}") P("\text{parents}" \vert "\text{my}") P("\text{!}" \vert "\text{parents}") P("\text{</s>}" \vert "\text{!}")$$



Here is a **trick** or **taste**! That is we augment the original word sequence with a start symbol $$ \text{<s>} $$ and an end symbol $$ \text{</s>} $$. So we can make every word in vocabulary to appear both at the left-hand and right-hand side of the conditional mark $$ \vert $$ which is a taste of statistical completeness. Since every word sequence will definitely start with a $$ \text{<s>} $$ symbol, so we know that $$ P("\text{<s>}") = 1$$. 

> **Statistical estimation.** Statistical estimation is to estimate the possible value or distribution of the model parameters. Their are many different flavor of statistical estimation. The main two categories are **point estimation** and **interval estimation** (a specific form of distribution estimation). Point estimation like maximum likelihood estimation (MLE), maximum posterior estimation (MAP) or Bayesian estimation by which the final estimation of the parameters is a point in the parameter's space. (Parameter space is a high-dimensional space $$\mathbb{R}^d$$, if we have $$ d $$ parameters in our model.) Each of which bears specific statistical properties. Interval estimation or distribution estimation is to maintain a distribution which we call **posterior** in Bayesian statistics over the parameter space $$ \mathbb{R}^d $$. Intuitively speaking, a distribution has all the benefits over point distribution since we could tune that distribution according to new data points, which is very suitable for online learning. 

Despite the benefits of Bayesian estimation, here we use maximum likelihood estimation to estimate a point value of those parameters of a language model, which is those conditional terms $$ P(\cdot \vert \cdot) $$. 

> **My comments.** Of course, the above paragraph does not say anything about the reason why we do not use other ways of parameter estimation. It is a good question to think about during discussion. 

#### Maximum likelihood estimation of LMs

If we do not use Markov assumption and parameterize the probability as following: 



$$ P(w_1, w_2, \dots, w_n) = P(w1) P(w_2 \vert w_1) \dots P(w_n \vert w_1, \dots, w_{n-1})$$



You can get very **inaccurate** estimation of each term $$ P(w_i \vert w_1, \dots, w_{i-1}) $$. Since a usual estimate is through counting, that is, to compute the frequency of every possible sequence of length $$i-1$$, $$ w'_1, \dots, w'_{i-1} $$ and the frequency of this specific pattern $$ w_1, \dots, w_i $$, and use the latter one to divide the former one, which is count ratio. Since it is **very seldom** that continuous word sequence $$ w_1, \dots, w_i $$ occurs very often when $$ i $$ is a big, so the count of specific $$w_1,\dots,w_n$$ is very small or most of them equal 0. 

Mathematically, this estimation can be written as: $$ P(w_i \vert w_1, \dots, w_{i-1}) \approx \frac{Count(w_1,\dots,w_i)}{\sum_{w'_1,\dots,w'_{i-1}}Count(w'_1,\dots,w'_{i-1})} $$.  

In Neubig's notes, he asked students to use **uni-gram** or **bi-gram** models to learn and estimate the parameters. 

- Uni-gram model is that we assume total independency between every two words $$ w_i $$ and $$w_j$$. So that the probability $$P(w_1,\dots,w_n)$$ can decompose to $$P(w_1) \cdot P(w_2) \cdot \dots \cdot P(w_n)$$. And what we should estimate is the 