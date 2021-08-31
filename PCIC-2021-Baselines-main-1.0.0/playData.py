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

user_num = 1000
item_num = 1720

P_L_TO = np.bincount(extract_alldata[:, 2], minlength=2)[:]
tmp = P_L_TO.sum()
P_L_TO = P_L_TO / P_L_TO.sum()

P_L_T = np.bincount(valid_data[:, 2], minlength=2)[:]
P_L_T = P_L_T / P_L_T.sum()

P_O_T = tmp / (user_num * item_num)
P = P_L_TO * P_O_T / P_L_T

propensity_score = [P] * item_num
inverse_propensity = np.reciprocal(propensity_score)
# print(P_L_TO)  # [0.7867772 0.2132228]
# print(P_L_T)  # [0.61794998 0.38205002]
# print(P_O_T)  # 0.011291279069767441
# print(P)  # [0.01437612 0.00630168]
# print(propensity_score)  # [array([0.01437612, 0.00630168]),...]

# 测试集中标签已知信息的个数
# sum_d = 0
# for i in range(test_data.shape[0]):
#     for j in range(extract_alldata.shape[0]):
#         if (test_data[i, 0] == extract_alldata[j, 0]) and (test_data[i, 1] == extract_alldata[j, 1]):
#             sum_d += 1
#
# print('the number of test\'s tags with known info:', sum_d)


# S_c 和 S_t 中重复的（用户，标签）个数
# sum_ct = 0
# for i in range(extract_bigtag.shape[0]):
#     for j in range(extract_choicetag.shape[0]):
#         if (extract_bigtag[i, 0] == extract_choicetag[j, 0]) and (extract_bigtag[i, 1] == extract_choicetag[j, 1]) and (extract_bigtag[i, 2] == extract_choicetag[j, 2]):
#             sum_ct += 1
#
# print('the number of the same info with S_c & S_t:', sum_ct)




