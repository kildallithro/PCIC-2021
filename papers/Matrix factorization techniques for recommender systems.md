# Matrix factorization techniques for recommender systems

## Recommender system strategies

一般来说推荐系统是基于以下两种策略之一：

1. **内容过滤**（the content filtering）：为每一个用户或者物品创造一个概要文件来描述其性质；
2. **协同过滤**（collaborative filtering）：基于用户过去的行为。分析用户之间的关系和产品之间的相互依赖关系来识别出新的 用户-物品 对。但协同过滤有一个问题就是冷启动(the cold start)问题。
   1. Neighborhood methods：以计算物品之间或者用户之间的关系为中心
   2. Latent factor models：尝试通过描述物品和用户来解释评级

## Matrix factorization methods

一些最成功的 Latent factor models 都是基于**矩阵分解**的。

我们将用户明确的反馈称为**评级**（rating），明确的反馈组成了一个稀疏矩阵，因为任何一个单一的用户只可能对很小比例的物品进行评级。

矩阵分解一个优势就是它允许额外的信息加入。当明确的反馈不可获得时，推荐系统可以用不明确的反馈来推断用户的偏好。

## A basic matrix factorization model

用户-物品 的交互可以被建模成这个空间中的**内积**。

目标函数：

$min_{q^*,p^*}\sum_{(u,i)\in\kappa}(r_{ui}-q_i^Tp_u)^2+\lambda(||q_i||^2+||p_u||^2)$ 

该公式中 $r_{ui}$ 是已知的（训练集）；防止过拟合加入了正则化项，其中 $\lambda$ 表示正则化程度；这里使用交叉验证（cross-validation）

## Learning algorithm

两种方法最小化上述方程：随机梯度下降法（stochastic gradient descent）和交替最小二乘（alternating least squares）

### Stochastic gradient descent

对于给定的训练集，系统预测 $r_{ui}$ ，并且计算相关的预测误差：

$e_{ui}=r_{ui}-q_i^Tp_u$

然后在梯度的反方向上以 $\gamma$ 大小的比例修改参数：

$q_i\leftarrow q_i+\gamma(e_{ui}p_u-\lambda q_i)$

$p_u\leftarrow p_u+\gamma(e_{ui}q_i-\lambda p_u)$

### Alternating least squares

由于 $q_i$ 和 $p_u$ 都是未知的，上述方程是非凸的。然而如果我们固定了其中一个位置参数，那么优化问题就变成了二次方程的，并且可以最优化解决的。

但是通常情况下，随机梯度下降法比 ALS 更容易且更快。两种情况下，ALS 会受欢迎：

1. 系统可以并行使用；
2. 针对以隐式数据为中心的系统，因为训练集不能被认为是稀疏的，所以像梯度下降那样在每个单独的训练案例上循环是不现实的。ALS 可以有效地解决这个问题。

## Adding biases

许多观察到的评价值变化是由于与用户或项目相关的影响，即偏见或截取，独立于任何交互。因此我们应该加入 biases。

对 $r_{ui}$ 进行评级所涉及到的一阶近似偏差为：

$b_{ui} = \mu + b_i + b_u$

其中 $\mu$ 表示的是总体的平均评级；参数 $b_i,b_u$ 分别表示用户 u 和物品 i 的观测偏差。

因此， 评级的估计可以写成：

$\hat{r}_{ui}=\mu+b_i+b_u+q_i^Tp_u$

因此该系统是通过最小化下面方程的最小误差来学习的：

$min_{p^*,q^*,b^*}\sum_{(u,i)\in \kappa}(r_{ui}-\mu-b_u-b_i-p_u^Tq_i)^2+\lambda(||p_u||^2+||q_i||^2+b_u^2+b_i^2)$

## Additional input sources

处理冷启动问题：推荐系统可以使用隐含的反馈来洞察用户的偏好。

$N(u)$ 表示用户 u 表达隐含偏好的一组物品集合。因此，用户表现出对 $N(u)$ 项的偏好是通过向量来表征的：

$\sum_{i\in N(u)}x_i$ 

归一化求和项是有好处的：

$|N(u)|^{-0.5}\sum_{i\in N(u)}x_i$

另一个信息源是已知用户的属性，例如，人口统计特征。用户 u 对应的属性集合为 $A(u)$，这个集合可以描述性别、年龄组、邮政编码、收入水平等等。一个显著的因子向量 $y_a\in \mathbb{R}^f$对应于每个属性，通过用户相关属性集合来描述一个用户:

$\sum_{a\in A(u)}y_a$

矩阵分解模型应该整合所有的信息源，来提高用户的表征：

$\hat{r}_{ui}=\mu+b_i+b_u+q_i^T[p_u+|N(u)|^{-0.5}\sum_{i\in N(u)}x_i+\sum_{a\in A(u)}y_a]$

## Temporal dynamics(实时动态)

事实上，产品的感知和受欢迎程度是会随着新选择的出现而发生改变的。

因此，评级的估计 $\hat{r}_{ui}=\mu+b_i+b_u+q_i^Tp_u$ 可以改写成：

$\hat{r}_{ui}=\mu+b_i(t)+b_u(t)+q_i^Tp_u(t)$

## Inputs with varying confidence levels

在一些设置中，并不是所有观察到的评级都值得相同的权重或置信度。

因此，矩阵分解模型可以很容易地接受不同的置信水平，这让它给更少意义的观察以更少的权重。如果观察到的 $r_{ui}$ 的置信度为 $c_{ui}$，那么该模型为了解释置信度增强了代价函数：

$min_{p^*,q^*,b^*}\sum_{(u,i)\in \kappa}c_{ui}(r_{ui}-\mu-b_u-b_i-p_u^Tq_i)^2+\lambda(||p_u||^2+||q_i||^2+b_u^2+b_i^2)$ 



























