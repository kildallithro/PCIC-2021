import numpy as np

from config import opt

user_num = 1000
item_num = 1720

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
rating_rate = np.loadtxt(dir + '/rating_rate.txt')


# 生成 rating_rate.txt
# rating = np.loadtxt(dir + '/rating.txt', dtype=int)
#
# rating_rate = rating[:, 2] / 5.0
# rating_rate = rating_rate.reshape(-1, 1)
# rating_rate = np.hstack((rating[:, 0: 2], rating_rate))
# print(rating_rate)
# np.savetxt("rating_rate.txt", rating_rate, fmt=('%d', '%d', '%.2f'))


# 生成 pro_score_rate.txt
dir = './data/train'
movie_data = np.loadtxt(dir + '/movie.txt', dtype=int)
rating_rate = np.loadtxt(dir + '/rating_rate.txt')

movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i, 1:]
    movie.append(tmp)

mat1 = np.zeros((user_num, item_num))
mat0 = np.zeros((user_num, item_num))
obs = np.zeros((user_num, item_num))
# print(movie[int(rating_rate[0, 1])])
for i in range(rating_rate.shape[0]):
    row = int(rating_rate[i, 0])
    if rating_rate[i, 2] > 0.7:
        for tag in movie[int(rating_rate[i, 1])]:
            mat1[row, tag] += rating_rate[i, 2]
            obs[row, tag] = 1
    elif rating_rate[i, 2] < 0.7:
        for tag in movie[int(rating_rate[i, 1])]:
            mat0[row, tag] += 1.4 - rating_rate[i, 2]
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
        P.append([0.999, 0.001])
    else:
        P.append([tmp0, tmp1])

P = np.array(P, dtype=float)
for j in range(item_num):
    P[j] = P[j] / P[j].sum()
    if P[j, 1] == 0.0:
        P[j, 0] = 0.999
        P[j, 1] = 0.001
    elif P[j, 0] == 0.0:
        P[j, 0] = 0.001
        P[j, 1] = 0.999

np.savetxt("./data/train/pro_score_rate.txt", P, fmt=('%f', '%f'))