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


# 生成 rating_rate.txt
rating = np.loadtxt(dir + '/rating.txt', dtype=int)

rating_rate = rating[:, 2] / 5
rating_rate = rating_rate.reshape(-1, 1)
rating_rate = np.hstack((rating[:, 0: 2], rating_rate))
print(rating_rate)
np.savetxt("./data/train/rating_rate.txt", rating_rate, fmt=('%d', '%d', '%.4f'))