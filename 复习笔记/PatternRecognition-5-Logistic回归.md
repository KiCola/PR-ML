---
title: PatternRecognition-5-Logistic回归
author: Zero 谔的薛定猫
date: 2025-01-03 16:35:36
tags: Pattern Recognition
categories: AI
cover: https://s2.loli.net/2024/01/01/5aAzeSl942kDdWn.jpg
mathjax: true
---

# 模式识别 U5 Logistic 回归

## 课堂内容

* 5.1  Logistic回归问题
* 5.2  Logistic回归损失
* 5.3  Logistic回归算法
* 5.4  二元分类线性模型讨论



### 5.1 逻辑斯蒂回归问题

<font color=brown>**逻辑斯蒂回归是一种“软分类”策略，即考虑分类中的概率性或称模糊性**</font>

逻辑回归假设数据服从伯努利分布，通过**极大似然函数**的方法，运用**梯度下降**来求解参数，来达到将**数据二分类**的目的

![image-20250103163801983](PatternRecognition-5-Logistic回归/image-20250103163801983.png)

![image-20250103163915226](PatternRecognition-5-Logistic回归/image-20250103163915226.png)

![image-20250103164138843](PatternRecognition-5-Logistic回归/image-20250103164138843.png)



### 5.2 逻辑斯蒂回归损失

往期损失函数回顾

![image-20250103164239805](PatternRecognition-5-Logistic回归/image-20250103164239805.png)



<font color=purple>**逻辑斯蒂回归可以使用平方损失函数作为损失函数吗？**</font>

#### 平方损失

数学推演：
$$
h(\mathbf x)=\theta(\mathbf w^T\mathbf x)=\frac{1}{1+e^{-\mathbf w^T\mathbf x}}\\
\\
\mathcal L_{in}(\mathbf w)=(\theta(\mathbf w^T\mathbf x)-y)^2\\
$$
存在一个问题：

![image-20250103165651372](PatternRecognition-5-Logistic回归/image-20250103165651372.png)

由实际训练样本标签带来的影响，我们改写损失函数形式为：
$$
\mathcal L_{in}(\mathbf w)=(\theta(y\mathbf w^T\mathbf x)-1)^2\\
$$
由此进行梯度推演：
$$
\begin{align*}
\frac{\partial \mathcal L_{in}(\mathbf w,\mathbf x,y)}{\partial w_i}&=2[\theta(y\mathbf w^T\mathbf x)-1]\frac{\partial\theta(\mathbf w)}{\partial w_i}\\
&=2[\theta(y\mathbf w^T\mathbf x)-1]\frac{yx_ie^{-y\mathbf w^T\mathbf x}}{(1+e^{-y\mathbf w^T\mathbf x})^2}\\
&=2[\theta(y\mathbf w^T\mathbf x)-1]yx_i\frac{1}{(1+e^{-y\mathbf w^T\mathbf x})}\frac{e^{-y\mathbf w^T\mathbf x}}{(1+e^{-y\mathbf w^T\mathbf x})}\\
&=2[\theta(y\mathbf w^T\mathbf x)-1]yx_i\theta(y\mathbf w^T\mathbf x)[1-\theta(y\mathbf w^T\mathbf x)]\\
\end{align*}

\theta(y\mathbf w^T\mathbf x)\gt0 (正确分类)\space \space \nabla\mathcal L \rightarrow 0\\
\theta(y\mathbf w^T\mathbf x)\lt0(错误分类)\space \space \nabla\mathcal L \rightarrow 0\\
$$
![image-20250103171707846](PatternRecognition-5-Logistic回归/image-20250103171707846.png)



#### 交叉熵损失

>![image-20250103172431435](PatternRecognition-5-Logistic回归/image-20250103172431435.png)
>
>![image-20250103172549248](PatternRecognition-5-Logistic回归/image-20250103172549248.png)![image-20250103172757562](PatternRecognition-5-Logistic回归/image-20250103172757562.png)

![image-20250103172208440](PatternRecognition-5-Logistic回归/image-20250103172208440.png)

![image-20250103192544657](PatternRecognition-5-Logistic回归/image-20250103192544657.png)

![image-20250103192715299](PatternRecognition-5-Logistic回归/image-20250103192715299.png)

![image-20250103192728885](PatternRecognition-5-Logistic回归/image-20250103192728885.png)

### 5.3 逻辑斯蒂回归算法

![image-20250103192818799](PatternRecognition-5-Logistic回归/image-20250103192818799.png)

![image-20250103192828708](PatternRecognition-5-Logistic回归/image-20250103192828708.png)

![image-20250103192837548](PatternRecognition-5-Logistic回归/image-20250103192837548.png)



### 5.4 二元分类线性模型讨论

![image-20250103192906149](PatternRecognition-5-Logistic回归/image-20250103192906149.png)

![image-20250103192927096](PatternRecognition-5-Logistic回归/image-20250103192927096.png)

![image-20250103193047418](PatternRecognition-5-Logistic回归/image-20250103193047418.png)

![image-20250103193110248](PatternRecognition-5-Logistic回归/image-20250103193110248.png)

![image-20250103193127083](PatternRecognition-5-Logistic回归/image-20250103193127083.png)



### 小结

![image-20250103193143557](PatternRecognition-5-Logistic回归/image-20250103193143557.png)