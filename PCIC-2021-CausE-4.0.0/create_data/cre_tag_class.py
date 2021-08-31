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


#==========================================================
# 生成关于标签的特征矩阵 (rating_rate.txt 和 movie.txt)
movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i, 1:]
    movie.append(tmp)

mat = np.zeros((1720, 2))
# i 表示电影id; mat 第一列表示 tag 在多少电影中出现过
for i in range(len(movie)):
    for tag in movie[i]:
        mat[tag][0] += 1
for i in range(rating_rate.shape[0]):
    row = int(rating_rate[i, 1])
    # mat[row][1] += rating_rate[i, 2]
    mat[row][1] += 1

# print(mat)
mat = np.array(mat, dtype=float)
np.savetxt("../data/train/tag_feature.txt", mat, fmt="%f ")

tag_feature = mat
# 设置gmm函数
gmm = GaussianMixture(n_components=opt.class_num, covariance_type='full').fit(tag_feature)
# 训练数据
y_pred = gmm.predict(tag_feature)
# # 带宽，也就是以某个点为核心时的搜索半径
# bandwidth = estimate_bandwidth(tag_feature, quantile=0.5)
# # 设置均值偏移函数
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# # 训练数据
# ms.fit(tag_feature)
# # 每个点的标签
# y_pred = ms.labels_
# # 总共的标签分类
# labels_unique = np.unique(y_pred)
# print(labels_unique)

# cnt = 0
# for i in range(len(y_pred)):
#     if y_pred[i] == y_pred[4]:
#         cnt += 1
# print(cnt)  # cnt==1
# print(y_pred)
tag_class = []
for i in range(opt.class_num):
    tag_class.append([])
    for j in range(len(y_pred)):
        if y_pred[j] == i:
            tag_class[i].append(j)
np.savetxt("../data/train/tag_class.txt", y_pred, fmt="%d")

# 绘图
# plt.scatter(mat[:, 0], mat[:, 1], c=y_pred)
# plt.show()


