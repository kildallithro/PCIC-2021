from config import opt
from utils import MF_DATA, CausE_DATA, evaluate_model
import pandas as pd
import numpy as np

dir = './data/train'
bigtag = np.loadtxt(dir + '/bigtag.txt', dtype=int)
choicetag = np.loadtxt(dir + '/choicetag.txt', dtype=int)
movie_data = np.loadtxt(dir + '/movie.txt', dtype=int)
rating = np.loadtxt(dir + '/rating.txt', dtype=int)
valid_data = np.loadtxt('./data/valid/validation.txt', dtype=int)
valid_P1_data = np.loadtxt('./data/valid/validation_P1.txt', dtype=int)
valid_P2_data = np.loadtxt('./data/valid/validation_P2.txt', dtype=int)
test_data = np.loadtxt('./data/test/test_P1.txt', dtype=int)
extract_alldata = np.loadtxt(dir + '/extract_alldata.txt', dtype=int)
extract_bigtag = np.loadtxt(dir + '/extract_bigtag.txt', dtype=int)
extract_choicetag = np.loadtxt(dir + '/extract_choicetag.txt', dtype=int)

# movie
movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i, 1:]
    movie.append(tmp)

tag_num = np.max(movie)
# print('tag_num:', tag_num+1)  # 1720
#
# print('test:', test_data.shape)  # (4454, 2)
# print('bigtag:', bigtag.shape)  # (8612, 3)
# print('choicetag:', choicetag.shape)  # (1540, 3)
# print('movie:', movie_data.shape)  # (1000, 9)
# print('rating:', rating.shape)  # (19903, 3)
#
# print('alldata_num:', extract_alldata.shape[0])  # 19421
# print('alldata_like:', np.count_nonzero(extract_alldata[:,2]))  # 4141
# print('bigtag_num:', extract_bigtag.shape[0])  # 14133
# print('bigtag_like:', np.count_nonzero(extract_bigtag[:,2]))  # 3889
# print('bigtag_like_pro:', np.count_nonzero(extract_bigtag[:,2]) / extract_bigtag.shape[0])  # 0.2751715842354773
# print('choicetag_num:', extract_choicetag.shape[0])  # 5802
# print('choicetag_like:', np.count_nonzero(extract_choicetag[:,2]))  # 558
# print('choicetag_like_pro:', np.count_nonzero(extract_choicetag[:,2]) / extract_choicetag.shape[0])  # 0.09617373319544985

# print(len(valid_data))  # 2039
# print(len(valid_P1_data))  # 1020
# print(len(valid_P2_data))  # 1019

# user_num = 1000
# item_num = 1720
#
# P_L_TO = np.bincount(extract_alldata[:, 2], minlength=2)[:]
# tmp = P_L_TO.sum()
# P_L_TO = P_L_TO / P_L_TO.sum()
#
# P_L_T = np.bincount(valid_data[:, 2], minlength=2)[:]
# P_L_T = P_L_T / P_L_T.sum()
#
# P_O_T = tmp / (user_num * item_num)
# P = P_L_TO * P_O_T / P_L_T
# P_ = np.bincount(extract_alldata[:, 2], minlength=2)[:] / ((user_num * item_num) * P_L_T)
#
# propensity_score = [P] * item_num
# inverse_propensity = np.reciprocal(propensity_score)
# print(P_L_TO)  # [0.7867772 0.2132228]
# print(P_L_T)  # [0.61794998 0.38205002]
# print(P_O_T)  # 0.011291279069767441
# print(P)  # [0.01437612 0.00630168]
# print(P_)  # [0.01437612 0.00630168]
# print(propensity_score)  # [array([0.01437612, 0.00630168]),...]
# print(inverse_propensity)  # [[ 69.55981399 158.68776677]...]

# 测试集中标签已知信息的个数
# sum_d = 0
# for i in range(test_data.shape[0]):
#     for j in range(extract_alldata.shape[0]):
#         if (test_data[i, 0] == extract_alldata[j, 0]) and (test_data[i, 1] == extract_alldata[j, 1]):
#             sum_d += 1
#
# print('the number of test\'s tags with known info:', sum_d)


# S_c 和 S_t 中重复的（用户，标签）个数
sum_ct = 0
for i in range(extract_alldata.shape[0]):
    for j in range(valid_P1_data.shape[0]):
        if (extract_alldata[i, 0] == valid_P1_data[j, 0]) and (extract_alldata[i, 1] == valid_P1_data[j, 1]):
            sum_ct += 1

print('the number of the same info with S_c & S_t:', sum_ct)


user_num = 1000
item_num = 1720

# mat_t = np.zeros((item_num, 2))
# for i in range(len(extract_alldata)):
#     if extract_alldata[i, 2] == 0:
#         mat_t[extract_alldata[i, 1]][0] += 1
#     elif extract_alldata[i, 2] == 1:
#         mat_t[extract_alldata[i, 1]][1] += 1
# P_L_TO = []
# mat_t = np.array(mat_t, dtype=float)
# for i in range(item_num):
#     if mat_t[i][0] == 0 and mat_t[i][1] == 0:
#         mat_t[i][0] = np.bincount(extract_alldata[:, 2], minlength=2)[0]
#         mat_t[i][1] = np.bincount(extract_alldata[:, 2], minlength=2)[1]
#     len1 = mat_t[i].sum()
#     P_L_TO.append([mat_t[i][0] / len1, mat_t[i][1] / len1])
# P_L_TO = np.array(P_L_TO)
# # print(P_L_TO)
#
# mat_v = np.zeros((item_num, 2))
# for i in range(len(valid_data)):
#     if valid_data[i, 2] == 0:
#         mat_v[valid_data[i, 1]][0] += 1
#     elif valid_data[i, 2] == 1:
#         mat_v[valid_data[i, 1]][1] += 1
# P_L_T = []
# tmp = []
# mat_v = np.array(mat_v, dtype=float)
# for i in range(item_num):
#     if mat_v[i][0] == 0 and mat_v[i][1] == 0:
#         mat_v[i][0] = np.bincount(valid_data[:, 2], minlength=2)[0]
#         mat_v[i][1] = np.bincount(valid_data[:, 2], minlength=2)[1]
#     len2 = mat_v[i].sum()
#     tmp.append(len2)
#     P_L_T.append([mat_v[i][0] / len2, mat_v[i][1] / len2])
# P_L_T = np.array(P_L_T)
# # print(P_L_T)
#
# tmp = np.array(tmp)
# P_O_T = tmp / (user_num * item_num)
# P = []
# for i in range(item_num):
#     P.append(P_L_TO[i] * P_O_T[i] / P_L_T[i])
# P = np.array(P)
# inverse_propensity = np.reciprocal(P)
# print(inverse_propensity)
#
# print(np.count_nonzero(mat_t), np.shape(mat_t))

# import torch
# import torch.nn as nn
#
# torch.manual_seed(2021)
# num_users = 1000
# num_items = 1720
# embedding_size = 16
#
# user = torch.LongTensor(test_data[:, 0]).to('cpu')
# item = torch.LongTensor(test_data[:, 1]).to('cpu')
#
# user_e = nn.Embedding(num_users, embedding_size)
# item_e = nn.Embedding(num_items, embedding_size)
# user_b = nn.Embedding(num_users, 1)
# item_b = nn.Embedding(num_items, 1)
#
# user_embedding = user_e(user)  # (4454, 16)
# item_embedding = item_e(item)  # (4454, 16)
#
# preds = user_b(user)
# preds += item_b(item)
# preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)
# preds = preds.squeeze()
# preds = preds.detach().cpu().numpy()
# # Normalize
# preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
# print(preds)


# rating.txt 中的信息
# cnt = 0
# dict = {}
# rate = []
# for i in range(bigtag.shape[0]):
#     for j in range(rating.shape[0]):
#         if bigtag[i, 0] == rating[j, 0] and bigtag[i, 1] == rating[j, 1]:
#             dict[bigtag[i, 0]] = bigtag[i, 1]
#             rate.append(rating[j, 2])
#             cnt += 1
# print(cnt)  # 29
# print(dict)
# print(rate)
# # [4, 4, 5, 5, 5, 3, 4, 4, 4, 5, 5, 5, 3, 3, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 4, 5, 4, 3]
#
# cnt_rate = {}
# for r in rating[:, 2]:
# 	# get(value, num) 函数的作用是获取字典中 value 对应的键值, num=0 指示初始值大小。
# 	cnt_rate[r] = cnt_rate.get(r, 0) + 1
# print(cnt_rate)  # {5: 5357, 4: 7738, 3: 5164, 2: 1398, 1: 246}


# laplace 平滑后的倾向得分概率矩阵
# train_data = np.loadtxt(opt.ps_train_data)
# train_data = train_data.astype(int)
# mat = np.zeros((user_num, item_num))
# obs = np.zeros((user_num, item_num))
# for i in range(train_data.shape[0]):
#     row = train_data[i, 0]
#     col = train_data[i, 1]
#     obs[row][col] = 1
#     if train_data[i, 2] == 1:
#         mat[row][col] = 1
# mat = np.array(mat, dtype=int)
#
# P = []
# for j in range(item_num):
#     tmp0 = 0
#     tmp1 = 0
#     for i in range(user_num):
#         if obs[i][j] == 1 and mat[i][j] == 1:
#             tmp1 += 1
#         if obs[i][j] == 1 and mat[i][j] == 0:
#             tmp0 += 1
#
#     P.append([tmp0, tmp1])
#
# P = np.array(P, dtype=float)
# for j in range(item_num):
#     P[j] = (P[j] + 1) / (P[j].sum() + 2)  # laplace 平滑
#
# print(P)


# like1 = np.bincount(extract_bigtag[:, 2])
# like2 = np.bincount(extract_choicetag[:, 2])
# print('dont like:', like1[0] / like1.sum(), like2[0] / like2.sum())




