---
layout: post
title: "PyTorch Basics Part 1 (in Chinese)"
author: Guanlin Li
tag: party
---

`2017.11.25 glli`

> 引子. 
>
> 一些经典的神经网络架构（前馈神经网、循环神经网、卷积神经网），或者根据使用者经验设计出的新的神经网络架构，我们将这些架构视为基本的计算模块，通过这些基本计算模块的组合、配搭，可以针对具体的机器学习任务（分类、回归、结构预测）设计出神经网络模型，通过训练数据得到一组使学习目标（函数）最优的模型参数，这样一个过程，叫做实用深度学习。如果读者希望总是使用深度学习的技术，上述的这一过程便会不断重复出现在读者的研究与实验活动中。
>
> 由于神经网络的学习，即参数的估计、学习目标的优化的过程，最常使用的是梯度下降（gradient descent）。梯度下降算法需要计算每一时刻，给定数据$$\{ x^i, y^i \}$$后，神经网络模型的误差函数的梯度，并沿着梯度的方向更新网络的参数：$$\theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L(\theta ; \{ x^i, y^i \})}$$。而如何计算神经网络参数的梯度，存在着名为“反向传播”的梯度计算算法，就如“给定函数，能通过求导法则去求出各变量的偏导数”一样，倘若我们能做到：给定神经网络模型由输入到输出，再到目标函数的这一计算流程，且能根据求导法则，计算出每个神经网络参数的偏导数（梯度）的话，这样的梯度计算与参数更新过程，就都能自动由软件替我们完成，我们所要做的，仅仅是指定神经网络的计算流程即可。将上述方案实现的软件，我们称之为深度学习框架（deep learning framework），这里要介绍的[PyTorch](http://pytorch.org/)便属于其中一个十分优秀的框架。

#### 1. 自动微分

自动微分（Automatic Differentiation）又叫算法微分（Algorithmic Differentiation），由来已久，目的是通过算法与软件自动化，简化人工通过求导法则计算复杂函数偏导数的代价；具体讲，自动微分能对给定的函数求一阶甚至高阶的偏导数，原理是：复合函数求导的基本法则，即链式法则。

复合函数求导，如：$$y=g \cdot f \cdot h(x)$$，对$$x$$求导，可以拆解为每个复合运算求导后再连乘的形式：

$$\partial{y}/\partial{x} = \partial{g}/\partial{f} \cdot \partial{f}/\partial{h} \cdot \partial{h}/\partial{x}$$

自动微分需要通过软件自动化的是：

1. 对一个函数，如何表示为一些简单运算的复合（往往以有向无环图的数据结构表示），并记录需要对哪些变量进行偏导的计算
2. 对每个简单运算，给定了输入输出，如何求偏导
3. 如何根据链式法则，将每一部分的偏导数衔接起来，对于需要求偏导的变量，得到完整的链式求导的结果

自动微分的软件在求偏导时，往往分为：前向模式（Forward mode）、反向模式（Backward mode）。我们同样以上述对于$$x$$求偏导为例，并且假设一些计算过程的中间量如下：

- $$a = h(x)$$
- $$b = f(a)$$
- $$y = g(b)$$

前向模式按照如下顺序去计算$$\partial{y}/\partial{x}$$，可以发现上一次的计算结果可以直接代入下一次的计算中：

- $$\partial{a}/\partial{x}$$
- $$\partial{b}/\partial{x} = \partial{b}/\partial{a} \cdot \partial{a}/\partial{x}$$
- $$\partial{y}/\partial(x) = \partial{y}/\partial(b) \cdot \partial{b}/\partial{x}$$

后向模式则相反，按照如下顺序去计算$$\partial{y}/\partial{x}$$，同样可以看出，上一次的计算结果被代入了下一次的计算中：

- $$\partial{y}/\partial{b}$$
- $$\partial{y}/\partial{a} = \partial{y}/\partial{b} \cdot \partial{b}/\partial{a}$$
- $$\partial{y}/\partial{x} = \partial{y}/\partial{a} \cdot \partial{a}/\partial{x}$$

> 思考. 
>
> 写到这里，希望读者能够结合上一次茶会最后关于自动求导的内容，思考一下，这一个自动求导的程序应该如何去写：1. 大概需要哪些数据结构去存储哪些量（有向无环图？有向无环图的每一个节点需要有哪些变量去存储哪些数据）；2. 前向模式和反向模式应该如何实现（有向图的拓扑序遍历？）

关于自动微分，上面的讲述十分粗糙，由于上次的茶会在最后部分大概阐述过，所以这里仅仅从概念与核心内容上进行了简单的强调，并没有举具体的例子。倘若读者对自动微分感兴趣，可以参见下面一些资料：

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf): 自动微分的一篇综述性文章；
- [Automatic Differentiation: or mathematically finding derivatives](http://www.columbia.edu/~ahd2125/post/2015/12/5/)：这是一篇博客性质的文章，通过例子写得十分清楚，并还恰当的启发了读着如何应用程序设计语言去实现，作者提到了[Introduction to AD](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/)，这篇博客，同样可以去看看。
- [Automatic differentiation and Backpropagation](https://www.suchin.co/2017/03/18/Automatic-Differentiation-and-Backpropagation/)：这篇博客结合了神经网络的自动求导，例子丰富，在文章最后给出了一个简单的前向网络用Python写出的一个自动求导的小程序，挺值得阅读的。

#### 2. 深度学习框架的基本组成

> 目的.
>
> 这一小节笔记的作用于目的在于，让读者对深度学习框架的架构有基本的了解，使得读者在接触新框架或阅读新框架代码时能够利用自己熟悉的框架的知识，进行迁移，并更快的掌握新框架。

所有现代深度学习框架所共有的两个特性是：

- **提供GPU并行计算的接口**：这里需要思考的问题是，为什么需要将神经网络的计算派发至GPU进行计算，传统的运算单元CPU的劣势在哪儿？
- **提供计算图构建与自动微分功能**：这一点想必读者已经耳熟能详了。

在华盛顿大学的深度学习系统课程中，另一个比较流行的深度学习框架[mxnet](https://mxnet.incubator.apache.org/)的作者之一陈天奇给出了一个“典型深度学习系统栈”的示意图：

![dlsys_arch](D:\wOrKsPaCe_old\Reading List\17\Tutorials\自然语言处理-晚茶会\imgs\pytorch\dlsys_arch.png)

从上图可以看出，一个框架的设计存在着许多需要考虑的问题：

- 上层的User API的设计往往可以参考现有的一些框架，theano与torch是两个相对较早的框架，它们有着各自优秀的特性，theano的**静态计算图与计算图优化**的设计思想启发了tensorflow与mxnet。后二者的核心：计算图构建与自动微分功能，均是基于静态图的。

- 静态图的优势在于，由于构建计算图时，须使用框架中自定义的流程控制逻辑（流程控制逻辑主要指：条件判断、循环）而非程序设计语言原生的控制逻辑（if、for、while等），所以更方便设计者设计对计算图进行程序分析，进而能够对计算图进行优化（例如：中间数据节点的创建、合并等）；

- 静态图的弊端在于：倘若模型的设计会考虑数据本身带有的内在结构（最好的例子是递归神经网中根据每个句子的句法树进行前向计算），即每一条数据的计算图可能都有差异，这样便无法事先统一构建出一个计算图，静态图遇到这类问题的解决方案是，对每一个样例都要进行分析，并构建一个静态图，由于每次构图都会进行图的优化，会一定程度降低训练时的计算效率，另外，静态图框架中自定义的流程控制逻辑不易理解与使用的弊端，也造成了许多刚接触深度学习的用户的一大困扰；

- 静态图的另一个弊端便是不方便用户使用程序设计语言原生的`print`进行调试：在theano中，**构图**即是将theano中的张量类型变量，不断进行theano内建的基本计算操作进行变换，从而得到目标变量的过程，最终会通过theano的函数算子进行编译得到一个函数，例如下面所示的sigmoid函数$$s = \frac{1}{1 + \exp(-X \cdot w)}$$的计算图的构建，其中$$X$$是数据矩阵，即X的每一行代表一个样例$$x \in \mathbb{R}^d$$。

  ```python
  import theano
  import theano.tensor as T
  X = T.dmatrix('X') # 创建一个double类型的数据矩阵，名为'X'
  w = T.dmatrix('w') # 创建一个double类型的参数矩阵，名为'w'
  y = T.dot(x, w) # X dot w，y可以看作是计算图的中间结果，因为我们可以直接写为 s = 1 / (1 + T.exp(-T.dot(X, w)))
  s = 1 / (1 + T.exp(-y)) # sigmoid函数，运算'/'，'+'，'T.exp'，'-'均是逐元素运算（element-wise op）
  logistic_func = theano.function([X, w], s) # 调用theano的function函数，对计算图进行优化分析，并得到一个可供调用的函数'logistic_func'，该函数的输入是'X'即一个double类型的矩阵，以及一个参数向量'w'，输出是's'即一个与'X'矩阵行数相同的向量s
  s_real = logistic_func(
      [[1.0, 1.0], [-1.0, -1.0]]
  ) # 得到的函数可以进行调用，输入为numpy类型或者python原生的list类型的变量
  print(s_real)
  ```

  倘若，我们希望或许计算过程的中间结果`y`，由于对计算图创建可调用函数时，`y`并非作为输出，所以我们无法访问得到`y`的值；只有显式的将`y`作为计算图的输出并构建可调用函数时，我们才能得到`y`的值，即：

  ```python
  logistic_func = theano.function([X, w], out=[s, y])
  s_real, y_real = logistic_func(
      [[1.0, 1.0], [-1.0, -1.0]], # X_real 真实数据
      [[0.0, 0.0]] # w_real 真实的参数矩阵
  )
  print(s_real)
  print(y_real)
  ```

  在theano和tensorflow中，通过其各自**自定义的op**得到的计算图，需要通过一个**容器**进行编译或运行，转换为可以接收**真实数据**进而进行计算得到**真实输出**的对象。在theano中需要通过`theano.function`进行显式的声明输入、输出，并得到一个函数对象，该函数对象即可以接收**与输入一致**的真实数据进行计算了；而在tensorflow中，需要通过调用`session.run()`进行真实数据的运算，如下面的相同功能的tensorflow代码：

  ```python
  import tensorflow as tf
  x_dim = 2
  X = tf.placeholder(tf.float32, [None, x_dim]) # 创建一个数据的占位符placeholder，需要制定其形状
  w = tf.Variable(tf.zeros([x_dim, 1])) # 创建一个参数矩阵，由于属于模型参数，所以通过tf.Variable创建为变量类型
  y = -tf.matmul(X, w) # [None, 1]
  s = 1 / (1 + tf.exp(y)) # [None, 1]的矩阵
  with tf.Session() as sess:
      tf.global_variables_initializer().run() # 执行所有计算图中Variable类型对象的初始化，这里即初始化参数向量'w'
      X_real = [[1.0, 1.0], [-1.0, -1.0]]
      s_real = sess.run(s, feed_dict={X: X_real}) # 计算's'，当给定输入是X_real时
      print(s_real)
  ```

  从上面的例子可以发现，在调用`sess.run(s, feed_dict={X: X_real})`时，类似于theano中创建函数，并给函数输入真实数据的过程，所以`sess.run()`函数不仅对计算图进行了优化，也同样执行了一次真实的计算。

- PyTorch和[Chainer](https://chainer.org/)的核心设计思想是基于动态图构建计算流程，用户可使用**程序设计语言原生的流程控制逻辑**进行计算图的构建，这部分的介绍在第4节中详细阐述。

- 在PyTorch中，用户所使用的API主要以Python包和模块的形式存在，经常使用的**包**包括：

  ![api](D:\wOrKsPaCe_old\Reading List\17\Tutorials\自然语言处理-晚茶会\imgs\pytorch\api.png)

  - `torch`：`torch`包，是一个张量运算的包，定义了各种基本运算、矩阵操作等；

  - `torch.Tensor`：`torch.Tensor`包，更准确的说是PyTorch所定义的一个类，作为数学概念中张量（tensor）的一个容器，提供了多种不同类型的张量的初始化方法以及计算，类似于matlab与numpy中的核心数据结构多维矩阵（multidimensional matrix）；

  - `torch.nn`：神经网络模块，**基本神经网络的架构**的实现，例如：实现了基本的前馈神经网（线性层+非线性激活），循环神经网，卷积神经网（卷积层、池化层）等的接口；以及常用的**损失函数**的实现；

    ![nn_module](D:\wOrKsPaCe_old\Reading List\17\Tutorials\自然语言处理-晚茶会\imgs\pytorch\nn_module.png)

  - `torch.optim`：最优化算法的实现，将基于梯度的最优化算法：随机梯度下降`torch.optim.SGD`，以及其几种变体`torch.optim.Adadelta`，`torch.optim.Adagrad`，`torch.optim.Adam`，`torch.optim.RMSprop`等；同时，实现了一个著名的拟牛顿算法`torch.optim.LBFGS`。

  - `torch.autograd`：自动求导模块，其中最重要的一个类是`Variable`，在使用时通过`from torch.autograd import Variable`加载到Python解释器中；该类可以通过封装`torch.Tensor`类型的变量，进行计算图的构建：即被`Variable`封装后的张量变量在进行前向计算时，`torch.autograd`模块会**跟踪**每一次基本计算Op（Operation，操作、基本计算），并动态构建计算图。

- 当然，上面所陈述的均为深度学习框架中用户接口层次的使用与设计，这里希望读者思考的是：**一个基于神经网络的学习/训练问题，其程序实现的流程大致是什么样的？**想明白了这个问题，就能对上述模块为什么要这么划分有更深入的理解。

- 顺着陈天奇的架构图往下走，就是系统级别的组件：包括计算图的优化与执行、以及运行时并行调度的功能。这部分是十分核心的，因为所有的OP最终都会被**链接**到基于GPU或者CPU开发的矩阵运算库的API中，并根据计算图的拓扑序进行前向或者反向计算，这部分计算发生在GPU的显存或CPU的缓存与内存中。由于GPU的计算核心数目远大于CPU，能通过并行存取、计算多个显存单元中的数据，所以特别适合矩阵运算与数据的批处理（batch learning），只要一批数据中的每一条数据的计算流程是一致的，就能够利用GPU进行批处理，计算效率会远高于CPU。

- GPU计算最常用的API是大名鼎鼎的英伟达公司（NVIDIA）设计开发的[CUDA](https://zh.wikipedia.org/wiki/CUDA)。更为准确的说，CUDA是一套英伟达设计的通用并行计算或通用GPU计算（[GPGPU](https://zh.wikipedia.org/wiki/%E5%9B%BE%E5%BD%A2%E5%A4%84%E7%90%86%E5%99%A8%E9%80%9A%E7%94%A8%E8%AE%A1%E7%AE%97)）架构，包括了用户接口与针对GPU硬件设配的编译器；深度学习主要使用CUDA提供的用户接口，将矩阵的基本运算（矩阵加法、乘法等）转换为CUDA中的相应计算，即可使用GPU的并行计算能力了。所以，几乎所有的依赖GPU的深度学习框架，都会在CUDA上进行封装，抽象出与上层用户API一致的OP接口以供上层Python包调用。另一方面，为了进行调度，框架设计者还会利用CUDA提供的并行计算组件管理GPU的内存，更适应于神经网络的计算特性。

  > 注记.
  >
  > 关于深度学习框架如何与CUDA以及cuDNN进行交互与衔接的内容，已经超出笔者的知识范围，但了解这部分内容是十分有价值的：GPU给了我们新一代数据处理范式，对GPU或并行异构编程（如OpenCL）的了解，就如同对分布式计算（如基于MapReduce）的了解一样同样重要。
  >
  > 在GPU程序设计中最终要的一环是对矩阵运算的优化，特别是矩阵乘法的优化，BLAS是一个线性代数基本运算标准，能够加速矩阵运算（与硬件计算资源CPU、GPU无关），[OpenBLAS](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)是中国学者张先轶主要维护的线性代数基本运算库，主要针对CPU开发；而英伟达也有着其对BLAS的GPU实现，称为[cuBLAS](http://docs.nvidia.com/cuda/cublas/index.html)，该API直接在CUDA之上实现，能访问GPU的计算资源。
  >
  > NVIDIA对于神经网络同样做了优化，在CUDA的基础上写了一个名为cuDNN的神经网络接口，其中高效实现了CNN与RNN
  >
  > 关于并行编程，Udacity上有一门课程，或许值得看一下——[并行编程入门](https://cn.udacity.com/course/intro-to-parallel-programming--cs344)。 

- PyTorch中我们如何将模型加载到GPU上进行计算呢？我们可以直接通过`.`运算符调用`cuda()`方法即可，在`torch.Tensor`包中的所有张量类，均实现了该方法，所以可以直接通过`t.cuda()`将张量`t`加载到GPU上。那么这里出现的一个疑问是，倘若实验的主机上有多个GPU，上述`t.cuda()`执行后，会将CPU中的张量`t`加载到哪一个GPU上呢？PyTorch中，对GPU基础管理模块叫`torch.cuda`，我们可以通过下面的程序来查看是否能访问GPU硬件设配，以及设置当前使用的GPU设备；当然我们还可以通过调用`t_cpu = t.cpu()`将`t`的GPU数据拷贝回CPU上的`t_cpu`变量中，但GPU上面的数据仍然存在：

  ```python
  import torch
  import torch.cuda as cuda
  t = torch.Tensor() # cpu上，内存中
  gpu_id = 0
  if cuda.is_available():
    cuda.set_device(gpu_id)
  t.cuda() # t在零号GPU上有了副本
  t_cpu = t.cpu() # 
  ```

  关于`cuda`类使用的语义，具体请参照API技术文档中关于[CUDA](http://pytorch.org/docs/0.2.0/) [Semantics](http://pytorch.org/docs/master/)这两部分，分别对应于版本号v0.2.0与v0.4.0a0（当前在master branch上的版本）。

  > 注意.
  >
  > 请读者重视CUDA Semantics这部分，这部分充分体现了深度学习框架是如何进行有效的内存管理与如何进行多GPU调度等功能。PyTorch中的cuda模块使用python编写，参见[这里](https://github.com/pytorch/pytorch/tree/master/torch/cuda)，通过调用`torch._C`类（该类位于[这里](https://github.com/pytorch/pytorch/tree/master/torch/csrc)）封装的cuda上下文管理接口进行GPU设备的管理。感兴趣的读者可以详细了解一下，并借此掌握一些GPU并行计算与CUDA编程的知识。

> 尾语.
>
> 通过上面的阐述，希望读者能够从框架整体设计与应用功能的角度上对PyTorch以及其他深度学习框架有一定的了解，逐渐习得这种模块化的认识，对于理解与使用新的框架（不局限于深度学习）是十分有好处的。



#### 3. PyTorch的基本数据结构——张量

张量是科学计算中数据的容器，在数据科学和数据时代中，张量是用于描述、分析数据的技术手段，[张量](http://www.offconvex.org/2015/12/17/tensor-decompositions/)[方法](https://simons.berkeley.edu/sites/default/files/docs/2930/slidesanandkumar.pdf)在机器学习中是兼具理论与实用性的优秀学习工具。在深度学习中，数据往往具有较高的维度，使用向量或者矩阵表示，举例子讲，做线性回归时，每一个输入样例是一个d维向量$$\mathbb{x} \in \mathbb{R}^d$$；而当一有L个词的句子的每个词表示为一个d维词向量时，这个句子就可以通过L个$$\mathbb{x}_i \in \mathbb{R}^d$$表示，例如可以组织成一个矩阵的形式：

$$X=[\mathbb{x_1, \mathbb{x}_2}, \dots, \mathbb{x}_L]_{d \times L}$$

回顾机器学习的有监督问题，无论是对于分类、回归还是结构预测，大多数情形下，都需要通过模型去建模一个输入到输出的映射这样的问题，在深度学习中，我们希望通过设计一个神经网络去实现这样的映射关系。要编程实现这样一个神经网络的学习问题，即我们要通过反向传播算法，去对给定的数据，计算神经网络在当前参数下的损失函数，并反向传播求得梯度，进而去更新网络的参数。

这一过程中，所有的数值都需要通过张量进行存储与管理，这些数值有：

- 模型的参数矩阵或向量
- 训练数据的数据矩阵或向量

在没有能实现自动微分的神经网络框架之前，人们通常采用matlab或者python的numpy包中的张量类进行上述数值的存储与运算。PyTorch的作者声称PyTorch用C++与Cython实现了类似于Numpy的张量类型、以及大部分的基本运算（Op，operation），我们下面来认识一下PyTorch的张量类型`torch.Tensor`。

![tensortype](D:\wOrKsPaCe_old\Reading List\17\Tutorials\自然语言处理-晚茶会\imgs\pytorch\tensortype.png)

从上表看出，torch实现了7种类型的张量，分别有CPU版本和GPU版本，当创建一个CPU张量对象后，可以通过执行其`.cuda`方法将该张量加载到指定的GPU设备上。上述类型中，使用最多的是`torch.FloatTensor`和`torch.LongTensor`，两个类：

```python
import torch
t_float_nodim = torch.FloatTensor() # 创建一个没有维度的张量
print(t_float_nodim.size())
print(type(t_float_nodim.size()))
# out
() # torch.Size 对象
torch.Size

t_float_vec = torch.FloatTensor(6) # 创建一个只有1维的张量——向量该维度上有6个元素
print(t_float_vec)
# out 
1.00000e-05 *
 -5.7287
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
[torch.FloatTensor of size 6]

print(t_float_vec.size())
# out
(6L,) # torch.Size 对象
```

使用这两个类最多的原因在于：

- 32-bit单精度浮点数的运算基本满足了机器学习算法中模型参数的精度，精度越低，显存占用越小，计算效率是越高的，所以有一些**片上固件式神经网络**采用的是16-bit甚至8-bit的浮点数；由于大部分GPU制造厂商生产的GPU对单精度浮点数运算有较好的支持，所以研究/开发人员在编写深度学习程序时，更常使用的数据类型是32-bit浮点数；
- 自然语言处理的研究/开发人员更多使用`torch.LongTensor`，是因为自然语言处理中，会有**词表**，词表中每个词汇对应一个embedding向量（可作为模型参数）；在一些开放域的NLP任务中，词表可能会达到百万量级，我们需要为此表中的每一个词分配一个ID号，即可通过`torch.LongTensor`进行存储。

#### 4. 神经网络的自动微分——`Variable`类、`autograd`模块

神经网络框架最大优势之一便是替我们自动计算神经网络参数的梯度。由于一个神经网络模型描述了从输入到输出的映射，该映射可以拆解为一个个基本计算的复合（类似于1中所述的函数的复合），所以，可以根据自动微分的原理与技术去实现神经网络的自动微分。

更广义的讲，所有深度学习框架均是一个自动微分的工具，均提供了大部分复合函数的自动求导机制。上句中“大部分”一词的意思在于：深度学习框架会尽可能地覆盖常用的基本运算，例如：初等运算中的加减乘除、指数对数、三角函数等运算。利用这些基本运算，使用户在使用时，可以构造出丰富的复合函数，以满足用户对计算流程的复杂需求。这些基本运算往往叫做“操作”或“算子”（Op, operation）。

前文曾提到过，`Variable`类封装的张量变量在进行前向计算时，会自动跟踪每一个基本计算，即`Variable`类至少含有的数据结构为下图所示，分别为`Variable`类型的成员变量：a). `Variable.data`, b). `Variable.grad`, c). `Variable.grad_fn`。

![Variable_class](D:\wOrKsPaCe_old\Reading List\17\Tutorials\自然语言处理-晚茶会\imgs\pytorch\Variable_class.png)

```python
import torch
t = torch.FloatTensor(4, 4).fill_(1) # 创建一个全1的4x4的FloatTensor
print(t)
# out
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
[torch.FloatTensor of size 4x4]

from torch.autograd import Variable
t_var = Variable(t)
print(t_var)
# out
Variable containing:
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
[torch.FloatTensor of size 4x4]

print(t_var.data) 
# out => 与t一致
print(t_var.data is t)
# out => True 即t_var.data与t引用相同，指向同一块内存/显存区域
print(t_var.grad_fn) # print(type(t_var.grad_fn))
# out
None
print(t_var.grad) # print(type(t_var.grad))
# out
None
```

下面我们通过一个一维的例子，实现函数：$$y = a^2+ b \cdot \exp(c)$$，并计算$$y$$对每一个自变量$$a, b, c$$的导数：

```python
import torch
from torch.autograd import Variable
a = Variable(torch.FloatTensor([1]).fill_(1))
b = Variable(torch.FloatTensor([1]).fill_(1))
c = Variable(torch.FloatTensor([1]).fill_(1))
a_square = a * a
bc_ret = b * torch.exp(c)
y = a_square + bc_ret

y.backward() # 通过调用Variable对象y的backward()方法，可以反向传播计算偏导数；但是该条语句会报错
# error
RuntimeError: there are no graph nodes that require computing gradients
```

该条错误告诉我们，通过`Variable`跟踪构建的计算图中没有任何一个节点需要对其进行求导，也就是说我们可以通过`Variable`的构造属性`requires_grad`设置创建的`Variable`是否需要在计算图中对其进行求梯度；显然我们这里的三个变量`a, b, c`均需要对其进行求导，所以上面代码应该改为：

```python
import torch
from torch.autograd import Variable
a = Variable(torch.FloatTensor([1]).fill_(1), requires_grad=True)
b = Variable(torch.FloatTensor([1]).fill_(1), requires_grad=True)
c = Variable(torch.FloatTensor([1]).fill_(1), requires_grad=True)
a_square = a * a
bc_ret = b * torch.exp(c)
y = a_square + bc_ret

y.backward()

print(a.grad)
# out
Variable containing:
 2
[torch.FloatTensor of size 1]

print(b.grad)
# out
Variable containing:
 2.7183
[torch.FloatTensor of size 1]

print(c.grad)
# out
Variable containing:
 2.7183
[torch.FloatTensor of size 1]

print(y.grad) # 由于y是输出，所以其梯度为0
# out
None
```
