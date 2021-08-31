# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'CausE'
    is_eval_ips = False

    data_dir = '/cache'

    train_data = data_dir + '/extract_alldata.txt'
    train_data_big = data_dir + '/extract_bigtag.txt'
    train_data_choice = data_dir + '/extract_choicetag.txt'
    val_all_data = data_dir + '/datasets/DatasetService/infer_valid/validation.txt'
    # test_data = data_dir + '/datasets/DatasetService/infer_test/test.txt'
    test_data = data_dir + '/datasets/DatasetService/infer_test/test_phase2.txt'


    # CausE data
    s_c_data = data_dir + '/extract_alldata.txt'  # 有偏的
    s_t_data = './data/valid/validation_P1.txt'  # 无偏的
    cause_val_data = './data/valid/validation_P2.txt'
    # s_c_data = data_dir + '/extract_bigtag.txt'  # 有偏的，观察数据
    # s_t_data = data_dir + '/extract_choicetag.txt'  # 无偏的，随机化实验数据
    # cause_val_data = './data/valid/validation.txt'

    reg_uc = 0.001
    reg_ut = 0.001
    reg_utc = 0.001

    reg_ic = 0.001
    reg_it = 0.001
    reg_itc = 0.001

    metric = 'auc'
    verbose = 10

    device = 'cpu'
    batch_size = 512
    embedding_size = 24

    max_epoch = 50  
    lr = 0.001 
    weight_decay = 1e-5

    pop_exp = 0.5
    class_num = 10
    pop_drift = 10


opt = DefaultConfig()
