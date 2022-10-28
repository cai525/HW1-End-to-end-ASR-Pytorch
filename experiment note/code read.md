## Note 

用于记录代码阅读的一些收获

### 01. log_softmax 

>  asr.py

[(17 封私信 / 30 条消息) log_softmax与softmax的区别在哪里？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/358069078)

pytorch中使用 log_softmax的好处在于防止溢出。

详情可参见李沐老师相关课程，以及上方的链接。



### 02. adadelta算法

> asr.py

参见： [Adadelta (keras.io)](https://keras.io/api/optimizers/adadelta/)

```python
tf.keras.optimizers.Adadelta(
    learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta", **kwargs
)
```

Optimizer that implements the Adadelta algorithm.

Adadelta optimization is a stochastic gradient descent method that is based on **adaptive learning rate per dimension** to address two drawbacks:

- The continual decay of learning rates throughout training.
- The need for a manually selected global learning rate.

Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta you don't have to set an initial learning rate. In this version, the initial learning rate can be set, as in most other Keras optimizers.



### 03. last_char

> asr.py -> ASR.forward

这是啥？

```python
decoder_input = torch.cat([last_char, context], dim=-1)
```

是将长句子上一次的最后一个char作为这次的首字符吗？



### 04. attention mechanism

> asr.py -> Attention.forward

attention这块不是很懂，改天再说吧...

### 05. bucket

> config 的yaml文件中

### 06. filter bank delta

大量实验表明，在语音特征中加入表征语音动态特性的差分参数，能够提高系统的识别性能。常用的是MFCC/filter bank参数的一阶差分参数(**Delta-Delta**)和二阶差分参数(**Delta-Delta+Delta**)。

### 07. Curriculum  

> config 的yaml文件中

[一篇综述带你全面了解课程学习(Curriculum Learning) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/362351969)