import numpy as np


user_num = 1000
item_num = 1720

dir = './data/train'
train_data = np.loadtxt(dir + '/extract_alldata.txt', dtype=int)
valid_data = np.loadtxt('./data/valid/validation.txt', dtype=int)
test_data = np.loadtxt('./data/test/test_P2.txt', dtype=int)

obs = np.zeros((user_num, item_num))
for i in range(train_data.shape[0]):
    for j in range(test_data.shape[0]):
        if train_data[i, 0] == test_data[j, 0] and train_data[i, 1] == test_data[j, 1]:
            row = test_data[j, 0]
            col = test_data[j, 1]
            obs[row][col] += 1

for i in range(valid_data.shape[0]):
    for j in range(test_data.shape[0]):
        if valid_data[i, 0] == test_data[j, 0] and valid_data[i, 1] == test_data[j, 1]:
            row = test_data[j, 0]
            col = test_data[j, 1]
            obs[row][col] += 1

print(obs.sum())
