@[TOC]
# 0 概述
&emsp;&emsp;在week 1中，我们学习了单变量线性回归，但现实世界往往不是简单的一元函数就能够表示的，因此基于Week 1的基础上，Week 2 将对多变量线性回归，进行深入的讲解。
&emsp;&emsp;并且通过调整学习率，特征缩放，特征构造，均值归一化，来使我们更好地进行迭代回归。

# 1. 课程大纲
本周课程大纲如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516110513392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
# 2. 课程内容
## 2.1 多元线性回归
&emsp;&emsp;在week 1中我们通过假设模型$h_{\theta}(x)=\theta_{0}+\theta_{1} x$去预测房价和面积之间的关系。但实际生活中房价不仅仅与面积有关，与新旧等多个因素相关，如下图所示：
![房价表](https://img-blog.csdnimg.cn/20190515223457913.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
决定房价的因素，从单变量变成了一个向量。
$x=\left[ \begin{array}{c}{x_{0}} \\ {x_{1}} \\ {x_{2}}\\... \\ {x_{n}}\end{array}\right]$
因此，假设模型就定义为：
$h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\cdots+\theta_{n} x_{n}$
这就是多变量线性回归。
**注**：这里需要说一下，什么样的函数可以被称为线性函数？
&emsp;&emsp;当函数满足以下两个性质，就被称为线性函数
&emsp;&emsp; 齐次性：$f(a x)=a f(x)$
&emsp;&emsp; 可加性：$f(x+y)=f(x)+f(y)$
## 2.2 多元线性回归的梯度下降
**假设函数**：
$h_{\theta}(x)=\theta^{T} x=\theta_{0} x_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\cdots+\theta_{n} x_{n}$
**损失函数**：
$J\left(\theta_{0}, \theta_{1}, \ldots, \theta_{n}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$
其中，$x^{(i)}$第i个训练数据的输入变量，现在是一个向量。
**优化目标**：
$\min _{\vec{\theta}} \mathrm{J}\left(\theta_{0}, \theta_{1},\cdots, \theta_{n}\right)$
**迭代步骤**：
$\theta_{j} :=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \ldots, \theta_{n}\right)$=$\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}$
直到收敛（**一般我们认为一次迭代${J}\left(\theta_{0}, \theta_{1},\cdots, \theta_{n}\right)$小于一个数值，比如10^-3，就可以认为收敛了**）。
&emsp;&emsp;**注意**：所有参数都要以同个costFunction当前参数来进行运算更新（不能更新一个$\theta_{j}$替换一个），最后替换生成新的costFunction。
## 2.3 特征缩放
&emsp;&emsp;从2.1节中的房价表，可以发现面积、房间数、年限都不在一个数量级上，这样会导致$\theta_{i}$的数值也不能在一个数量级，最终导致不能够简单快速的收敛。
&emsp;&emsp;因此，提出了特征缩放来解决这个问题。
**特征缩放**：将所有的特征变量压缩到大致为[-1, 1]的范围。
比如：
$x_{i}=\frac{x_{i}}{s_{i}}$ 其中，$s_{i}$是最大值-最小值，这样就缩放到[0-1]范围。
&emsp;&emsp;或者采用均值归一化，
$x_{i}=\frac{x_{i}-\mu_{i}}{s_{i}}$， 其中，$\mu_{i}$是$X$向量元素的平均值，$s_{i}$是最大值-最小值或者可以用标准偏差。
&emsp;&emsp;通过特征缩放，能够使得训练过程更加有效的收敛。
## 2.4 学习率
&emsp;&emsp;每一次迭代更新$\theta$，都会用到$\alpha$，它被称为学习率。一般情况下：
$\alpha$过小：收敛速度过慢
$\alpha$过大：不能收敛，甚至可能发散，有时候也会出现收敛过慢。
&emsp;&emsp;一般$\alpha$可以从 0.001，0.003，0.01，0.03，0.1，0.3，1 这几个值去尝试，选一个最优的。

## 2.5 构造特征以及多项式回归
&emsp;&emsp;到目前为止，我们仅仅通过图表知道房价和面积，年代，房间数...特征有关系，但是并不明确知道为什么会是这些特征，是否还有其他特征。往往如果这些简单的特征无法正确表达房价，是否能够尝试构造一个特征呢？
答案是肯定的。
&emsp;&emsp;比如构造特征$x_{1} / x_{2}$，表示房间总面积除以房间数，表征每个房间的平均面积，作为新的特征，或者对面积进行开方的到新的特征。
$h_{\theta}(x)=\theta_{0}+\theta_{1}(s i z e)+\theta_{2} \sqrt{(s i z e)}$

**关于如何去选取和构造特征**：这只能通过观察和推测了，需要在相关领域有足够的经验。在机器学习中，有专门的特征工程，就是为了能够找到比较好的feature。

## 2.6 正规方程法
### 2.6.1 正规方程的推导
&emsp;&emsp;在初中的时候，学习解方程组，老师常说：有几个方程就能解出几个变量，其实那就是正规方程的开端。然而真实情况并非如此，这里必须有一个前提，所有的方程的系数是线性无关的。
&emsp;&emsp;现在引入的正规方程，同样是根据初中时候的思想，不用不断迭代，一次就能够求解出最优的$\theta$。
&emsp;&emsp;如下图所示，我们引入$x_{0}$列，作为$\theta_{0}$，也就是偏移量固定系数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190516082047143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NTRE5fU1VTQU4=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;对于矩阵 $X$的每一行，相当于一个训练数据的输入变量，记为$x^{(i)}$，
$x^{(i)}=\left[ \begin{array}{c}{x_{0}^{(i)}} \\ {x_{1}^{(i)}} \\ {\vdots} \\ {x_{n}^{(i)}}\end{array}\right]$，其中i表示第i个训练数据，$i \in[1, m]$； $n$表示输入变量$x^{(i)}$第$n$个特征；$x^{(i)}$有n+1维。
因此可以将矩阵$X$表示为：
$X=\left[ \begin{array}{c}{(x^{(1)}})^{T} \\ ({x^{(2)}})^{T} \\ {\vdots} \\ {(x^{(m)})^{T}}\end{array}\right]$，可以看到矩阵$X$是$m*(n+1)$维的。
可以将向量$\theta$表示为：
$\Theta=\left[ \begin{array}{c}{\theta_{0}} \\ {\theta_{1}} \\ {\vdots} \\ {\theta_{n}}\end{array}\right]$
可以将输出向量$Y$表示为：
$Y=\left[ \begin{array}{c}{y^{(1)}} \\ {y^{(2)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right]$
综上可以得到正规方程组：
**假设模型**：
$h_{\Theta}(X)=X*\Theta$
**损失函数**：
$\mathrm{F}(\Theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$，用矩阵$X$和向量$\Theta$，$Y$，简化损失函数，
$X \cdot \Theta-Y=\left[ \begin{array}{c}\\ {\left(x^{(1)}\right)^{T}} \\ {\vdots} \\ {\left(x^{(m)}\right)^{T}}\end{array}\right] \cdot \left[ \begin{array}{c}{\theta_{0}} \\ {\theta_{1}} \\ {\vdots} \\ {\theta_{n}}\end{array}\right]-\left[ \begin{array}{c}{y^{(1)}} \\ {y^{(2)}} \\ {\vdots} \\ {y^{(m)}}\end{array}\right]=\left[ \begin{array}{c}{h_{\theta}\left(x^{(1)}\right)-y^{(1)}} \\ {h_{\theta}\left(x^{(2)}\right)-y^{(2)}} \\ {\vdots} \\ {h_{\theta}\left(x^{(m)}\right)-y^{(m)}}\end{array}\right]$
因此，可简化损失函数为：
$\mathrm{F}(\Theta)=\frac{1}{2 m}(X \cdot \Theta-Y)^{T}(X \cdot \Theta-Y)$
这就是矩阵和向量优雅的表示。
**经过推到，可得结论**：
$\Theta=\left(X^{T} X\right)^{-1} X^{T} Y$
其中前提条件是 $X^{T}X$ 是非奇异(非退化)矩阵， 即$\left|X^{T} X\right| != 0$
### 2.6.2 梯度下降和正规方程的比较
**梯度下降法**，
&emsp;&emsp;优点：梯度下降在超大数据集的训练中，表现依然良好
&emsp;&emsp;缺点：需要选择学习率$\alpha$，需要很多次迭代计算才能收敛
**正规方程法**，
&emsp;&emsp;优点：不用多次迭代，不用选择学习率$\alpha$
&emsp;&emsp;缺点：需要计算 $X^{T}X$，且其时间复杂度为$O\left(n^{3}\right)$，一般情况下，当n>10000的时候，就不考虑使用正规方程了。；无法计算Logistic Regression 的classification问题
## 2.7 Octave 简明教程
&emsp;&emsp;本课程所有的代码，都是使用octave完成的，octave可以理解为对Matlab的GUN免费版，基本的语法都和Matlab一致。当然如果你有盗版的Matlab，也可以直接使用。不过由于语法的些许不同，可能会耽误调试和提交代码，Octave同样很好用建议直接下载使用。
&emsp;&emsp;Octave教程内容较多，我将单独写一篇关于octave的总结。

# 3. 课后编程作业
&emsp;&emsp;我将课后编程作业的参考答案上传到了github上，包括了octave版本和python版本，大家可参考使用。

# 4. 总结
&emsp;&emsp;多变量线性回归以及多项式回归，让我们进一步的对复杂的回归问题有了了解。
&emsp;&emsp;同时学习工具octave也可以帮助我们快速的完成编码实现回归的工作。
&emsp;&emsp;在以后的工作中，octave会成为有力的工具，因为它能够快速实现你的想法，当你实现了这些想法并发现有效后，再用java或者c或者python做进一步部署，这将是一个很好的工作思路。

















