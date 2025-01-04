---
title: PatternRecognition 1 绪论
author: Zero 谔的薛定猫
date: 2025-01-01 21:21:19
tags: 
 - Pattern Recognition
categories: 
 - AI
cover: https://s2.loli.net/2024/01/01/5aAzeSl942kDdWn.jpg
---

# 模式识别 U1 绪论

## 课程内容

### 模式与模式识别

<font color=brown>**什么是模式？**</font>

* “模式是混沌的对立面，他是一个可赋予名字、无确切定义的实体。”
  “A pattern is the opposite of a chaos; it is an entity vaguely defined , that could be given a name.” —— Satoshi Watanabe
* “模式是由确定性和随机性组成的一组对象，过程或事件。”
  “A pattern is a set of objects, processes or events which consists of both deterministic and stochastic components.”
* …

模式是对客观对象的描述

<font color=red>**模式识别是通过使用计算机算法来自动发现数据中的规律性，并应用这些规律性来作出决策，例如将数据分类到不同的类别中。**</font>



**模式识别的三项主要任务**

* **表征** Representation
  * 如何表示对象类别
* **学习** Learning
  * 给定训练数据如何生成分类器
* **识别** Recognition
  * 对未见过的数据实现分类



### 模式识别与机器学习方法

模板匹配 Template Matching

目标识别对于计算机是一件困难的事

简单的模板匹配方法不能找到感兴趣的目标

挑战：

1. 视点
2. 光照
3. 遮挡
4. 尺度
5. 变形
6. 背景混杂
7. 类内差异大
8. 类间差异小



**生成式方法 Generative**

![image-20250101213954104](PatternRecognition-1-绪论/image-20250101213954104.png)

**判别式方法 Discriminant **

![image-20250101214021132](PatternRecognition-1-绪论/image-20250101214021132.png)



**分类器的设计**

![image-20250101214114523](PatternRecognition-1-绪论/image-20250101214114523.png)

![image-20250101214140310](PatternRecognition-1-绪论/image-20250101214140310.png)



**算法模型大纲**

* 感知器
* 线性回归
* Fisher线性判别
* 逻辑斯蒂回归
* 非线性变换
* 线性支撑向量机
* 对偶SVM与核SVM
* 多类分类
* 神经网络与深度学习
* 卷积神经网络
* 集成学习
* 统计决策方法
* 概率密度函数的参数估计
* 概率密度函数的非参数估计



**损失函数**

* 0-1损失
* L1损失
* L2损失
* 交叉熵损失
* …



**参数优化**

* 梯度下降法
* 随机梯度下降法
* 批量随机梯度下降法
* 动量法
* …



**分类性能指标**

构建混淆矩阵

![image-20250101214516030](PatternRecognition-1-绪论/image-20250101214516030.png)

![image-20250101214713387](PatternRecognition-1-绪论/image-20250101214713387.png)

