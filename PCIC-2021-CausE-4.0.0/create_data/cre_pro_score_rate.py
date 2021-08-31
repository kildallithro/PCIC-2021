import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth

from config import opt

user_num = 1000
item_num = 1720

dir = '../data/train'
bigtag = np.loadtxt(dir + '/bigtag.txt', dtype=int)
choicetag = np.loadtxt(dir + '/choicetag.txt', dtype=int)
movie_data = np.loadtxt(dir + '/movie.txt', dtype=int)
rating = np.loadtxt(dir + '/rating.txt', dtype=int)
valid_data = np.loadtxt('../data/valid/validation.txt', dtype=int)
valid_P1_data = np.loadtxt('../data/valid/validation_P1.txt', dtype=int)
valid_P2_data = np.loadtxt('../data/valid/validation_P2.txt', dtype=int)
test_data = np.loadtxt('../data/test/test_P1.txt', dtype=int)
extract_alldata = np.loadtxt(dir + '/extract_alldata.txt', dtype=int)
extract_bigtag = np.loadtxt(dir + '/extract_bigtag.txt', dtype=int)
extract_choicetag = np.loadtxt(dir + '/extract_choicetag.txt', dtype=int)
rating_rate = np.loadtxt(dir + '/rating_rate.txt')
tag_class = np.loadtxt(dir + '/tag_class.txt')


#==========================================================
# 生成 pro_score_rate.txt (rating_rate.txt 和 tag_class.txt)
cnt_rate = {}
for r in rating_rate[:, 2]:
	# get(value, num) 函数的作用是获取字典中 value 对应的键值, num=0 指示初始值大小。
	cnt_rate[r] = cnt_rate.get(r, 0) + 1
values = list(cnt_rate.values())
values = np.array(values)
rate = values / values.sum()
print(rate)


# 去除电影第 1 列的标签列
movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i, 1:]
    movie.append(tmp)

# 根据用户对电影的评分，利用该电影对应的 8 个标签，计算出每个标签的倾向得分
mat1 = np.zeros((user_num, item_num))
mat0 = np.zeros((user_num, item_num))
obs = np.zeros((user_num, item_num))
# print(movie[int(rating_rate[0, 1])])
threshold = 0.7  # 0.7
for i in range(rating_rate.shape[0]):
    row = int(rating_rate[i, 0])
    if rating_rate[i, 2] > threshold:
        for tag in movie[int(rating_rate[i, 1])]:
            mat1[row, tag] = max(mat1[row, tag], rating_rate[i, 2])
            # mat1[row, tag] += rating_rate[i, 2]
            obs[row, tag] = 1
    elif rating_rate[i, 2] < threshold:
        for tag in movie[int(rating_rate[i, 1])]:
            mat0[row, tag] += max(mat0[row, tag], (1.2 - rating_rate[i, 2]))  #
            # mat0[row, tag] += (2 * threshold - rating_rate[i, 2])
            obs[row, tag] = 1

P = []
for j in range(item_num):
    cnt = 0
    tmp1 = 0
    tmp0 = 0
    for i in range(user_num):
        if obs[i][j] == 1:
            tmp1 += mat1[i][j]
            tmp0 += mat0[i][j]
            cnt += 1
    if cnt == 0:
        # P.append([0.999999, 0.000001])
        P.append([1.0, 0.0])
    else:
        P.append([tmp0, tmp1])

P = np.array(P, dtype=float)
P_copy = np.array(P, dtype=float)

# 一个向量对应一类标签
tag_class_list = []
for i in range(opt.class_num):
    tag_class_list.append([])
    for j in range(len(tag_class)):
        if tag_class[j] == i:
            tag_class_list[i].append(j)

for j in range(item_num):
    P[j] = P[j] / P[j].sum()

# 处理缺失值
for j in range(item_num):
    if P[j, 0] == 0.0:
        cls = tag_class[j]  # 标签 j 属于的类别
        cls_list = tag_class_list[int(cls)]
        P[j, 0] = P[cls_list, 0].sum() / len(cls_list)
        P[j, 1] = P[cls_list, 1].sum() / len(cls_list)
    if P[j, 1] == 0.0:
        P[j, 0] = 1 - opt.miss_val
        P[j, 1] = opt.miss_val



np.savetxt("../data/train/pro_score_rate.txt", P, fmt=('%f', '%f'))

