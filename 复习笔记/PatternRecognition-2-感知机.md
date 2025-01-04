---
title: PatternRecognition-2-感知机
author: Zero 谔的薛定猫
date: 2025-01-01 22:34:17
tags: Pattern Recognition
categories: AI
cover: https://s2.loli.net/2024/01/01/5aAzeSl942kDdWn.jpg
mathjax: true
---

# 模式识别 U2 感知机

## 课堂内容

### 感知器模型参数空间

感知器 Perceptron

![image-20250102081522037](PatternRecognition-2-感知机/image-20250102081522037.png)

![image-20250102081557112](PatternRecognition-2-感知机/image-20250102081557112.png)

![image-20250102081632039](PatternRecognition-2-感知机/image-20250102081632039.png)



#### 用向量形式（Vector Form）来表示感知器模型

$$
\begin{align*}
h(x) &= sign((\Sigma^d_{i=1}w_ix_i)-threshold)\\
     &= sign((\Sigma^d_{i=1}w_ix_i)+(-threshold)·(+1))\\
     &= sign(\Sigma^d_{i=0}w_ix_i)\\
     &= sign(\overrightarrow{w^T}\cdot \overrightarrow{x})
\end{align*}
$$

<font color=red>由上式我们可知，我们将阈值threshold扩展进了原来的w权重向量中，使其作为常数偏置存在；在进行这一操作时也在X中扩展出了一维全为1的增广X</font>

我们称新的w，X为增广化后的$\mathbf{w}^T$、$\mathbf X$
$$
\begin{align*}
\mathbf W &=[w_1, w_2, w_3,..., w_d,w_{d+1}]_{1×(d+1)}\\
\overrightarrow{x_i} &=[x_1, x_2, x_3,..., x_d, 1]_{1×(d+1)}\\
\mathbf X &=\begin{bmatrix}
\overrightarrow{x_1}\\
\overrightarrow{x_2}\\
\overrightarrow{x_3}\\
...\\
\overrightarrow{x_n}\\
\end{bmatrix}_{n×(d+1)}\\
则\mathbf X \cdot \mathbf W^T &=\begin{bmatrix}
\overrightarrow{x_1}\cdot \mathbf W^T\\
\overrightarrow{x_2}\cdot \mathbf W^T\\
\overrightarrow{x_3}\cdot \mathbf W^T\\
...\\
\overrightarrow{x_n}\cdot \mathbf W^T\\
\end{bmatrix}_{n×1}\\
&=\begin{bmatrix}
x_1^{(1)}w_1+x_2^{(1)}w_2+...+x_d^{(1)}w_d+1^{(1)}\cdot(w_{d+1})\\
x_1^{(2)}w_1+x_2^{(2)}w_2+...+x_d^{(2)}w_d+1^{(2)}\cdot(w_{d+1})\\
x_1^{(3)}w_1+x_2^{(3)}w_2+...+x_d^{(3)}w_d+1^{(3)}\cdot(w_{d+1})\\
...\\
x_1^{(n)}w_1+x_2^{(n)}w_2+...+x_d^{(n)}w_d+1^{(n)}\cdot(w_{d+1})\\
\end{bmatrix}_{n \times 1}

\end{align*}
\\ \mathbf W 是 1×(d+1)维, \mathbf X是 n×(d+1)维
$$


![image-20250102082706890](PatternRecognition-2-感知机/image-20250102082706890.png)

![image-20250102085809594](PatternRecognition-2-感知机/image-20250102085809594.png)

![image-20250102085843878](PatternRecognition-2-感知机/image-20250102085843878.png)

**在高维空间中感知器的分类面**
$$
\begin{align*}
h(x)&=sign(w_0+w_1x_1+w_2x_2+...+w_dx_d)\\
&=sign(\sum^{d}_{i=0}w_ix_i)\\
&=sign(W\cdot \bold{x})
\end{align*}
$$
![image-20250102092740345](PatternRecognition-2-感知机/image-20250102092740345.png)

![image-20250102092751229](PatternRecognition-2-感知机/image-20250102092751229.png)

![image-20250102092803212](PatternRecognition-2-感知机/image-20250102092803212.png)

>几何知识：二维中 点到直线的距离
>$$
>l:ax_1+bx_2+c=0\\
>则距离 \\
>d=\frac{|ax_{p1}+bx_{p2}+c|}{\sqrt{a^2+b^2}}
>$$
>扩展到如今向量几何当中
>$$
>d = r\frac{\mathbf{w}}{||\mathbf{w}||}\\
>其中\frac{\mathbf{w}}{||\mathbf{w}||}表示单位法向量，r则可以用标量指示距离\\
>P.||\mathbf{w}||指向量的模，也可理解为向量\mathbf{w}的L_2范数
>$$

![image-20250102092814413](PatternRecognition-2-感知机/image-20250102092814413.png)

有上述推导我们可得：
$$
r=\frac{g(\mathbf{x})}{||\mathbf{w}||}
$$
其中

$\mathbf{w}是训练得到的感知器模型，其本质是可学习迭代的参数集合；\\g(\mathbf{x})则是将该数据点代入模型中取得的结果$



<font color=red>**$W^T\cdot X=||W||*||X||cos\theta$**</font>



### 感知器算法 PLA

PLA（Perceptron Learning Algorithm）

#### 算法思路

![image-20250102165138225](PatternRecognition-2-感知机/image-20250102165138225.png)

![image-20250102165224918](PatternRecognition-2-感知机/image-20250102165224918.png)

![image-20250102165232635](PatternRecognition-2-感知机/image-20250102165232635.png)

![image-20250102165252113](PatternRecognition-2-感知机/image-20250102165252113.png)

#### 算法流程

![image-20250102165306904](PatternRecognition-2-感知机/image-20250102165306904.png)

![image-20250102201612008](PatternRecognition-2-感知机/image-20250102201612008.png)

#### 算法迭代示例

<img src="PatternRecognition-2-感知机/image-20250102170111825.png" alt="image-20250102170111825" style="zoom: 33%;" /><img src="PatternRecognition-2-感知机/image-20250102170130335.png" alt="image-20250102170130335" style="zoom:33%;" /><img src="PatternRecognition-2-感知机/image-20250102170225566.png" alt="image-20250102170225566" style="zoom:33%;" /><img src="PatternRecognition-2-感知机/image-20250102170214794.png" alt="image-20250102170214794" style="zoom:33%;" /><img src="PatternRecognition-2-感知机/image-20250102170238810.png" alt="image-20250102170238810" style="zoom:33%;" /><img src="PatternRecognition-2-感知机/image-20250102170250087.png" alt="image-20250102170250087" style="zoom:33%;" />

#### 算法问题

![image-20250102170517060](PatternRecognition-2-感知机/image-20250102170517060.png)

### 感知器算法的收敛性

<font color=red>**PLA收敛条件：数据集中所有样本线性可分**</font>

![image-20250102170637279](PatternRecognition-2-感知机/image-20250102170637279.png)

<font color=green>**所有样本线性可分是否意味着PLA一定收敛？**</font>

![image-20250102170924824](PatternRecognition-2-感知机/image-20250102170924824.png)

![image-20250102171010251](PatternRecognition-2-感知机/image-20250102171010251.png)

![image-20250102171030088](PatternRecognition-2-感知机/image-20250102171030088.png)

### 线性不可分情况

#### 线性不可分分析

![image-20250102171112008](PatternRecognition-2-感知机/image-20250102171112008.png)

对于**线性可分**情况

![image-20250102171222977](PatternRecognition-2-感知机/image-20250102171222977.png)

模型的最终目的是实现收敛，即全部样本完全正确划分



对于**线性不可分**情况

![image-20250102171201028](PatternRecognition-2-感知机/image-20250102171201028.png)

调整模型算法停止条件为:损失函数最小



#### Pocket算法

为处理线性不可分情况而对PLA算法的修正

![image-20250102171433089](PatternRecognition-2-感知机/image-20250102171433089.png)

![image-20250102171534230](PatternRecognition-2-感知机/image-20250102171534230.png)

### 小结

![image-20250102171600054](PatternRecognition-2-感知机/image-20250102171600054.png)

## 作业部分

### 手写作业

![image-20250102171807106](PatternRecognition-2-感知机/image-20250102171807106.png)

![img](PatternRecognition-2-感知机/d8d0a9fa7e87c4d48cbe078b2f5bd56c_720.png)

![image-20250102173714156](PatternRecognition-2-感知机/image-20250102173714156.png)

![img](PatternRecognition-2-感知机/21be8b82304e7c32a215f7af9ed9b43c.png)



![image-20250102174119401](PatternRecognition-2-感知机/image-20250102174119401.png)

![img](PatternRecognition-2-感知机/f4cd3be2992901bdf6c398bc024007bd_720.png)

![image-20250102192623875](PatternRecognition-2-感知机/image-20250102192623875.png)

![img](PatternRecognition-2-感知机/5b5b09b735fee4d5358f889e8c44568a_720.png)

### 编程作业

