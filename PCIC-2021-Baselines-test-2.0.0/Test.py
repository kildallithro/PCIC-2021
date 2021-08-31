import torch
import torch.nn as nn
import numpy as np

# 建立词向量层，次数为 13，嵌入向量维数为 3
embed = nn.Embedding(13, 3, padding_idx=0)

# 句子对['I am a boy.','How are you?','I am very lucky.']
# batch = [['i','am','a','boy','.'],['i','am','very','lucky','.'],['how','are','you','?']]

# 将batch中的单词词典化，用index表示每个词（先按照这几个此创建词典）
# batch = [[2,3,4,5,6],[2,3,7,8,6],[9,10,11,12]]

# 每个句子实际长度
# lens = [5,5,4]

# 加上EOS标志且index=0
# batch = [[2,3,4,5,6,0],[2,3,7,8,6,0],[9,10,11,12,0]]

# 每个句子实际长度（末端加上EOS）
lens = [6, 6, 5]

# PAD 过后，PAD 标识的 index=1
batch = [[2, 3, 4, 5, 6, 0], [2, 3, 7, 8, 6, 0], [9, 10, 11, 12, 0, 1]]

# RNN 的每一步要输入每个样例的一个单词，一次输入 batch_size 个样例
# 所以 batch 要按 list 外层是时间步数(即序列长度)，list 内层是 batch_size 排列。
# 即 [seq_len,batch_size]
batch = np.transpose(batch)

batch = torch.LongTensor(batch)

embed_batch = embed(batch)
print(embed_batch)


# TODO: i want to do

# time() 的应用
# from time import time
# t1 = time()
# for i in range(100000):
#     if i % 10000 == 0:
#         print('%.7f' % time())
#         print('%.7f' % (time() - t1))

a = [[1,2,3],
     [3,4,5],
     [4,5,7]]

print(np.max(a))
print(np.count_nonzero(a))

