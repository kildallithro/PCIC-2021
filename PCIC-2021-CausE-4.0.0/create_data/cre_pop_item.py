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
# test_data = np.loadtxt('../data/test/test_P1.txt', dtype=int)
test_data2 = np.loadtxt('../data/test/test_P2.txt', dtype=int)
extract_alldata = np.loadtxt(dir + '/extract_alldata.txt', dtype=int)
extract_bigtag = np.loadtxt(dir + '/extract_bigtag.txt', dtype=int)
extract_choicetag = np.loadtxt(dir + '/extract_choicetag.txt', dtype=int)
rating_rate = np.loadtxt(dir + '/rating_rate.txt')


#========================================================
# 根据 rating 数据集求每个 item 被访问的频率
movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i, 1:]
    movie.append(tmp)

obs = np.zeros((user_num, item_num))
print(movie[int(rating_rate[0, 1])])
for i in range(rating_rate.shape[0]):
    row = int(rating_rate[i, 0])
    for tag in movie[int(rating_rate[i, 1])]:
        obs[row, tag] = 1
for i in range(extract_alldata.shape[0]):  #
    row = int(extract_alldata[i, 0])
    col = int(extract_alldata[i, 1])
    obs[row, col] = 1
obs = np.array(obs, dtype=int)

pop_item = np.zeros((item_num, 1))
for j in range(item_num):
    n = np.bincount(obs[:, j])
    if n[0] == 1000:
        pop_item[j] = 1e-6
    else:
        pop_item[j] = n[1] / np.sum(n)
        # n1 = obs[:, j].sum()
        # pop_item[j] = n1 / np.sum(obs)
# print(obs)
# Normalization
pop_item = (pop_item - np.min(pop_item)) / (np.max(pop_item) - np.min(pop_item))
np.savetxt("../data/train/pop_item.txt", pop_item, fmt="%f")


#========================================================
# item 在验证集上的受欢迎程度
obs_val = np.zeros((user_num, item_num))
for i in range(valid_P2_data.shape[0]):  #
    row = int(valid_P2_data[i, 0])
    col = int(valid_P2_data[i, 1])
    obs_val[row, col] = 1
obs_val = np.array(obs_val, dtype=int)

pop_item_val = np.zeros((item_num, 1))
for j in range(item_num):
    n = np.bincount(obs_val[:, j])
    if n[0] == 1000:
        pop_item_val[j] = 1e-6
    else:
        pop_item_val[j] = n[1] / np.sum(n)
        # n1 = obs_val[:, j].sum()  #
        # pop_item_val[j] = n1 / np.sum(obs_val)
# print(obs)
pop_item_val = (pop_item_val - np.min(pop_item_val)) / (np.max(pop_item_val) - np.min(pop_item_val))
np.savetxt("../data/valid/pop_item_val.txt", pop_item_val, fmt="%f")


#=============================================================
# item 在测试集上的受欢迎程度
# obs_tst = np.zeros((user_num, item_num))
# for i in range(test_data.shape[0]):
#     row = int(test_data[i, 0])
#     col = int(test_data[i, 1])
#     obs_tst[row, col] += 1
# obs_tst = np.array(obs_tst, dtype=int)
#
# pop_item_tst = np.zeros((item_num, 1))
# for j in range(item_num):
#     n = np.bincount(obs_tst[:, j])
#     if n[0] == 1000:
#         pop_item_tst[j] = 1e-6
#     else:
#         n1 = obs_tst[:, j].sum()  #
#         pop_item_tst[j] = n1 / np.sum(obs_tst)
# # print(obs)
# pop_item_tst = (pop_item_tst - np.min(pop_item_tst)) / (np.max(pop_item_tst) - np.min(pop_item_tst))
# np.savetxt("../data/test/pop_item_tst.txt", pop_item_tst, fmt="%f")


# obs_tst = np.zeros((user_num, item_num))
# for i in range(test_data2.shape[0]):
#     row = int(test_data2[i, 0])
#     col = int(test_data2[i, 1])
#     obs_tst[row, col] += 1
# obs_tst = np.array(obs_tst, dtype=int)
# 
# pop_item_tst = np.zeros((item_num, 1))
# for j in range(item_num):
#     n = np.bincount(obs_tst[:, j])
#     if n[0] == 1000:
#         pop_item_tst[j] = 1e-6
#     else:
#         n1 = obs_tst[:, j].sum()  #
#         pop_item_tst[j] = n1 / np.sum(obs_tst)
# # print(obs)pop_item_tst2.txt
# pop_item_tst = (pop_item_tst - np.min(pop_item_tst)) / (np.max(pop_item_tst) - np.min(pop_item_tst))
# np.savetxt("../data/test/", pop_item_tst, fmt="%f")