---
title: PatternRecognition-3-LinearRegression
author: Zero 谔的薛定猫
date: 2025-01-02 19:39:56
tags: Pattern Recognition
categories: AI
cover: https://s2.loli.net/2024/01/01/5aAzeSl942kDdWn.jpg
mathjax: true
---

# 模式识别 U3 线性回归 Linear Regression

## 课程内容

* 3.1 线性回归问题
* 3.2 线性回归算法
* 3.3 梯度下降法



### 线性回归问题

机器学习的过程其实是一个找**最拟合函数**的过程，通过不断的训练，我们最终得到一个函数映射，给定函数（网络）一个输入，函数（网络）会给出相应的输出

**若输出的是一个数值（scatter），我们就将这类机器学习问题称为回归Regression**

![image-20250102194854398](PatternRecognition-3-LinearRegression/image-20250102194854398.png)



### 线性回归算法

#### 模型构建

![image-20250102194920951](PatternRecognition-3-LinearRegression/image-20250102194920951.png)

为达到回归目的，我们度量模型输出结果时不再仅仅关注输出的符号

而是关注模型输出的数值

<font color=brown>**由此，我们择取 “平方误差函数”作为我们的损失函数，来度量我们的模型学习效果**</font>
$$
\begin{align*}
\mathcal{L}&=(\hat{y_n}-y_n)^2 \\
\mathcal{L_{in}}&=\frac{1}{N}\sum^N_{n=1}(\hat{y_n}-y_n)^2
\end{align*}
$$
$\mathcal{L}_{in}$计算训练样本集所有样本产生的平均损失
$$
\mathcal{L_{in}}(h)=\frac{1}{N}\sum_{n=1}^N(h(\mathbf{x}_n)-y_n)^2
$$
$h(\mathbf{x}_n)$是回归模型的结果

<font color=blue>**在线性回归模型中，$\hat{y_n} = h(\mathbf{x}_n) = \mathbf{w}^T \mathbf{x}_n$**</font>
$$
\mathcal{L_{in}}(\mathbf{w})=\frac{1}{N}\sum_{n=1}^N(\mathbf{w}^T\mathbf x_n-y_n)^2
$$
则
$$
g=\arg \mathop{\min}\limits_{\mathbf w}\frac{1}{N}\sum_{n=1}^N(\mathbf{w}^T\mathbf x_n-y_n)^2
$$

#### 向量/矩阵形式

![image-20250102200406091](PatternRecognition-3-LinearRegression/image-20250102200406091.png)

![image-20250102200515442](PatternRecognition-3-LinearRegression/image-20250102200515442.png)

#### 广义逆解法——线性回归的解析解

$\nabla \mathcal{L}_{in}(\mathbf{w})=0$时，求得最佳解$\mathbf w^*$

![image-20250102200923255](PatternRecognition-3-LinearRegression/image-20250102200923255.png)

![image-20250102200957769](PatternRecognition-3-LinearRegression/image-20250102200957769.png)

![image-20250102201030278](PatternRecognition-3-LinearRegression/image-20250102201030278.png)

### 梯度下降法 Gradient Descent

#### 各种 GD算法

![image-20250102201512905](PatternRecognition-3-LinearRegression/image-20250102201512905.png)

![image-20250102201632439](PatternRecognition-3-LinearRegression/image-20250102201632439.png)

![image-20250102201641380](PatternRecognition-3-LinearRegression/image-20250102201641380.png)

![image-20250102201710172](PatternRecognition-3-LinearRegression/image-20250102201710172.png)

![image-20250102201737568](PatternRecognition-3-LinearRegression/image-20250102201737568.png)

##### <font color=red>**梯度下降法**</font>

![image-20250102202107696](PatternRecognition-3-LinearRegression/image-20250102202107696.png)

![image-20250102202158583](PatternRecognition-3-LinearRegression/image-20250102202158583.png)

![image-20250102203846428](PatternRecognition-3-LinearRegression/image-20250102203846428.png)

![image-20250102203857141](PatternRecognition-3-LinearRegression/image-20250102203857141.png)



##### <font color=red>AdaGrad 自适应梯度下降法</font>>

![image-20250102203919184](PatternRecognition-3-LinearRegression/image-20250102203919184.png)

![image-20250102203933820](PatternRecognition-3-LinearRegression/image-20250102203933820.png)

![image-20250102204102009](PatternRecognition-3-LinearRegression/image-20250102204102009.png)

![image-20250102204157937](PatternRecognition-3-LinearRegression/image-20250102204157937.png)

![image-20250102204311191](PatternRecognition-3-LinearRegression/image-20250102204311191.png)

##### <font color=red>RMSProp</font>

![image-20250102204400731](PatternRecognition-3-LinearRegression/image-20250102204400731.png)

![image-20250102204427507](PatternRecognition-3-LinearRegression/image-20250102204427507.png)



<font color=purple>**问题2：梯度为0就能得到全局最优解吗？  ×**</font>

![image-20250102204535320](PatternRecognition-3-LinearRegression/image-20250102204535320.png)

##### <font color=brown>Momentum 动量法梯度下降</font>

![image-20250102204637972](PatternRecognition-3-LinearRegression/image-20250102204637972.png)

![image-20250102204643736](PatternRecognition-3-LinearRegression/image-20250102204643736.png)

![image-20250102204728183](PatternRecognition-3-LinearRegression/image-20250102204728183.png)

![image-20250102204806891](PatternRecognition-3-LinearRegression/image-20250102204806891.png)



##### Adam 亚当优化器

![image-20250102204834359](PatternRecognition-3-LinearRegression/image-20250102204834359.png)

![image-20250102204914034](PatternRecognition-3-LinearRegression/image-20250102204914034.png)



<font color=purple>**问题3：训练样本批量大小的影响？**</font>

### batch_size 的影响

![image-20250102205002145](PatternRecognition-3-LinearRegression/image-20250102205002145.png)

![image-20250102205011908](PatternRecognition-3-LinearRegression/image-20250102205011908.png)

![image-20250102205057883](PatternRecognition-3-LinearRegression/image-20250102205057883.png)

![image-20250102205251315](PatternRecognition-3-LinearRegression/image-20250102205251315.png)

![image-20250102205328063](PatternRecognition-3-LinearRegression/image-20250102205328063.png)

![image-20250102205350908](PatternRecognition-3-LinearRegression/image-20250102205350908.png)

![image-20250102205412584](PatternRecognition-3-LinearRegression/image-20250102205412584.png)

![image-20250102205439768](PatternRecognition-3-LinearRegression/image-20250102205439768.png)

![image-20250102205457908](PatternRecognition-3-LinearRegression/image-20250102205457908.png)

![image-20250102205522457](PatternRecognition-3-LinearRegression/image-20250102205522457.png)



### 小结

![image-20250102205539604](PatternRecognition-3-LinearRegression/image-20250102205539604.png)

## 作业

### 手写作业

![image-20250102205639481](PatternRecognition-3-LinearRegression/image-20250102205639481.png)

![img](PatternRecognition-3-LinearRegression/7f8ffc6a6aafa1fc7cdad60812b75e04_720.png)