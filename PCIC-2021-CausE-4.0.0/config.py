# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'CausE'
    is_eval_ips = False

    data_dir = './data/'

    train_data = data_dir + '/train/extract_alldata.txt'
    train_data_big = data_dir + '/train/extract_bigtag.txt'
    train_data_choice = data_dir + '/train/extract_choicetag.txt'
    val_all_data = data_dir + '/valid/validation.txt'
    # test_data = data_dir + '/test/test_P1.txt'
    test_data = data_dir + '/test/test_P2.txt'


    # CausE data
    s_c_data = data_dir + '/train/extract_alldata.txt'  # 有偏的
    s_t_data = data_dir + '/valid/validation_P1.txt'  # 无偏的
    cause_val_data = data_dir + '/valid/validation_P2.txt'
    # s_c_data = data_dir + '/train/extract_alldata.txt'  # 有偏的
    # s_t_data = data_dir + '/train/extract_choicetag.txt'  # 无偏的
    # cause_val_data = data_dir + '/valid/validation.txt'

    reg_uc = 0.001
    reg_ut = 0.001
    reg_utc = 0.001  # not used

    reg_ic = 0.001
    reg_it = 0.001
    reg_itc = 0.001

    metric = 'auc'
    verbose = 10

    device = 'cpu'
    batch_size = 512
    embedding_size = 24

    max_epoch = 60
    lr = 0.001
    weight_decay = 1e-5

    pop_exp = 0.5
    class_num = 10
    pop_drift = 10
    miss_val = 1e-6  # 1e-3


opt = DefaultConfig()
