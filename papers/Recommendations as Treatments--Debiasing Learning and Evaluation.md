# Recommendations as Treatments: Debiasing Learning and Evaluation

针对选择偏差（selection bias）提出的解决方法。

方法的实现：https://www.cs.cornell.edu/~schnabts/mnar/

## Unbiased performance estimation for recommendation

### 任务1：估计等级预测的准确度

$\hat{R}_{naive}(\hat{Y})=\frac{1}{|{(u,i):O_{u,i}=1}|}\sum_{(u,i):O_{u,i}=1}\delta_{u,i}(Y,\hat{Y})$           (1)

我们把这个叫做 naive estimator。在选择偏差下，$\hat{R}_{naive}(\hat{Y})$ 对于真实的 $R(\hat{Y})$ 并不是无偏估计的。

### 任务2：评估推荐的质量

相比于评估预测评级的准确性，我们可能希望更直接地评估特定推荐的质量

$[\hat{Y}_{u,i}=1]\Leftrightarrow$ i  is recommended to u

CG（the Cumulative Gain）：$\delta_{u,i}(Y,\hat{Y})=(I/k)\hat{Y}_{u,i}\cdot Y_{u,i}$ 

DCG（the Discounted Cumulative Gain）：$\delta_{u,i}(Y,\hat{Y})=(I/log(rank(\hat{Y}_{u,i})))\cdot Y_{u,i}$

PREC@k（Precision at k）：$\delta_{u,i}(Y,\hat{Y})=(I/k)Y_{u,i}\cdot \mathbb{1}\{rank(\hat{Y}_{u,i})\leq k\}$ 

### 基于倾向得分的性能估计

**IPS Estimator（The Inverse-Propensity-Scoring）**：

$\hat{R}_{IPS}(\hat{Y}|P)=\frac{1}{U\cdot I}\sum_{(u,i):O_{u,i}=1}\frac{\delta_{u,i}(Y,\hat{Y})}{P_{u,i}}$           (2)

IPS 估计器对于任何分配机制都是无偏的

**SNIPS Estimator（The Self-Normalized Inverse-Propensity-Scoring）**：

$\hat{R}_{SNIPS}(\hat{Y}|P)=\frac{\sum_{(u,i):O_{u,i}=1}\frac{\delta_{u,i}(Y,\hat{Y})}{P_{u,i}}}{\sum_{(u,i):O_{u,i}=1}\frac{1}{P_{u,i}}}$           (3)

这种可以减少可变性的技术是利用控制变量的。















