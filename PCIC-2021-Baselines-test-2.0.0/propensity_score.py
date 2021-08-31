import numpy as np
from config import opt


user_num = 1000
item_num = 1720


dir = './data/train'
pro_score_rate = np.loadtxt(dir + '/pro_score_rate.txt')


def pro_score_like(ps_data):
    mat = np.zeros((user_num, item_num))
    obs = np.zeros((user_num, item_num))
    for i in range(ps_data.shape[0]):
        row = ps_data[i, 0]
        col = ps_data[i, 1]
        obs[row][col] = 1
        if ps_data[i, 2] == 1:
            mat[row][col] = 1
    mat = np.array(mat, dtype=int)

    P = []
    for j in range(item_num):
        tmp0 = 0
        tmp1 = 0
        cnt = 0
        for i in range(user_num):
            if obs[i][j] == 1 and mat[i][j] == 1:
                tmp1 += 1
                cnt += 1
            if obs[i][j] == 1 and mat[i][j] == 0:
                tmp0 += 1
                cnt += 1
        if cnt == 0:
            P.append([0.999, 0.001])
        else:
            P.append([tmp0, tmp1])

    P = np.array(P, dtype=float)

    # 将概率为 0 的行重新赋值，使得求倒数时概率不为无穷大
    for j in range(item_num):
        P[j] = P[j] / P[j].sum()
        if P[j, 0] == 1.0 and P[j, 1] == 0.0:
            P[j, 0] = 0.999
            P[j, 1] = 0.001
        elif P[j, 0] == 0.0 and P[j, 1] == 1.0:
            P[j, 0] = 0.001
            P[j, 1] = 0.999

    return P


# propensity estimation for MF_IPS
def cal_propensity_score(model_name):
    if model_name == 'MF_IPS':
        threhold = 0.99  # 可改进
        train_data_choice = np.loadtxt(opt.ps_train_data_choice)
        train_data_choice = train_data_choice.astype(int)
        # train_data_big = np.loadtxt(opt.ps_train_data_big)
        # train_data_big = train_data_big.astype(int)
        val_data = np.loadtxt(opt.ps_val_data)
        val_data = val_data.astype(int)

        P_L_TOC = pro_score_like(train_data_choice)
        # P_L_TOB = pro_score_like(train_data_big)
        # P_L_TO = threhold * P_L_TOC + (1 - P_L_TOC) * P_L_TOB
        P_L_TO = P_L_TOC
        P_L_TR = pro_score_rate
        P_L_T = pro_score_like(val_data)

    elif model_name == 'CausE':
        raw_matrix_c = np.loadtxt(opt.s_c_data)
        raw_matrix_t = np.loadtxt(opt.s_t_data)
        train_data = np.vstack((raw_matrix_c, raw_matrix_t))
        train_data = train_data.astype(int)
        val_data = np.loadtxt(opt.cause_val_data)
        val_data = val_data.astype(int)

        P_L_TO = pro_score_like(train_data)
        P_L_TR = pro_score_rate
        P_L_T = pro_score_like(val_data)
    else:
        return []

    # propensity_score = P_L_TO / (P_L_T * user_num * item_num)
    propensity_score = (P_L_TO + P_L_TR) / (2 * P_L_T * user_num * item_num)

    return propensity_score
