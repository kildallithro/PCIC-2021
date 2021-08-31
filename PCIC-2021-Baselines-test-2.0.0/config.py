# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'
    is_eval_ips = False

    data_dir = './data/'

    train_data = data_dir + '/train/extract_alldata.txt'
    train_data_big = data_dir + '/train/extract_bigtag.txt'
    train_data_choice = data_dir + '/train/extract_choicetag.txt'
    val_all_data = data_dir + '/valid/validation.txt'
    test_data = data_dir + '/test/test_P1.txt'

    # IPS data
    ps_train_data = data_dir + '/train/extract_alldata.txt'
    ps_train_data_big = data_dir + '/train/extract_bigtag.txt'
    ps_train_data_choice = data_dir + '/train/extract_choicetag.txt'
    ps_val_data = data_dir + '/valid/validation.txt'

    # CausE data
    s_c_data = data_dir + '/train/extract_alldata.txt'  # 有偏的
    s_t_data = data_dir + '/valid/validation_P1.txt'  # 无偏的
    cause_val_data = data_dir + '/valid/validation_P2.txt'
    # s_c_data = data_dir + '/train/extract_bigtag.txt'  # 有偏的，观察数据
    # s_t_data = data_dir + '/train/extract_choicetag.txt'  # 无偏的，随机化实验数据
    # cause_val_data = data_dir + '/valid/validation.txt'


    reg_c = 0.001
    reg_t = 0.001
    reg_tc = 0.001

    metric = 'mse'
    verbose = 50

    device = 'cpu'
    batch_size = 512
    embedding_size = 16

    max_epoch = 50  # Best Epoch: about 17
    lr = 0.001 
    weight_decay = 1e-5

opt = DefaultConfig()
