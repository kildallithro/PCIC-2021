import numpy as np
from config import opt


user_num = 1000
item_num = 1720


dir = './data/train'
pro_score_rate = np.loadtxt(dir + '/pro_score_rate.txt')
tag_class = np.loadtxt(dir + '/tag_class.txt')


def pro_score_like(ps_data):
    mat = np.zeros((user_num, item_num))
    for i in range(ps_data.shape[0]):
        row = ps_data[i, 0]
        col = ps_data[i, 1]
        if ps_data[i, 2] == 1:
            mat[row][col] = 1
        else:
            mat[row][col] = -1
    mat = np.array(mat, dtype=int)

    P = []
    for j in range(item_num):
        tmp0, tmp1, cnt = 0, 0, 0
        for i in range(user_num):
            if mat[i][j] == 1:
                tmp1 += 1
                cnt += 1
            elif mat[i][j] == -1:
                tmp0 += 1
                cnt += 1
        # cnt==0 表示没有用户对这个标签作出反馈
        if cnt == 0:
            # P.append([0.999999, 0.000001])
            P.append([1.0, 0.0])
        else:
            P.append([tmp0, tmp1])

    P = np.array(P, dtype=float)

    tag_class_list = []
    for i in range(opt.class_num):
        tag_class_list.append([])
        for j in range(len(tag_class)):
            if tag_class[j] == i:
                tag_class_list[i].append(j)

    for j in range(item_num):
        P[j] = P[j] / P[j].sum()

    # 将概率为 0 的行重新赋值，使得求倒数时概率不为无穷大
    for j in range(item_num):
        if (P[j, 0] == 1.0 and P[j, 1] == 0.0) or (P[j, 0] == 1.0 and P[j, 1] == 0.0):
            cls = tag_class[j]  # 标签 j 属于的类别
            cls_list = tag_class_list[int(cls)]
            P[j, 0] = P[cls_list, 0].sum() / len(cls_list)
            P[j, 1] = P[cls_list, 1].sum() / len(cls_list)

    return P


# propensity estimation for CausE
def cal_propensity_score():
    threshold_c = 0.9  # 0.9
    threshold_t = 0.9  # 0.9

    # model_name == 'CausE':
    # raw_matrix_c = np.loadtxt(opt.s_c_data)
    # raw_matrix_t = np.loadtxt(opt.s_t_data)
    # train_data = np.vstack((raw_matrix_c, raw_matrix_t))
    train_data_c = np.loadtxt(opt.s_c_data)
    train_data_c = train_data_c.astype(int)
    train_data_t = np.loadtxt(opt.s_t_data)
    train_data_t = train_data_t.astype(int)
    val_data = np.loadtxt(opt.cause_val_data)
    val_data = val_data.astype(int)

    P_L_TO_c = pro_score_like(train_data_c)
    P_L_TO_t = pro_score_like(train_data_t)
    P_L_TR = pro_score_rate
    P_L_T = pro_score_like(val_data)

    propensity_score_c = (threshold_c * P_L_TO_c + (1 - threshold_c) * P_L_TR) / P_L_T
    propensity_score_t = (threshold_t * P_L_TO_t + (1 - threshold_t) * P_L_TR) / P_L_T

    return propensity_score_c, propensity_score_t
