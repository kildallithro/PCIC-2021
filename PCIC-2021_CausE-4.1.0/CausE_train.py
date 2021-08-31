import os
import models
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from time import time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
from torch.nn.init import normal_
from sklearn.metrics import roc_auc_score
import copy

from config import opt
from propensity_score import cal_propensity_score
from metrics import AUC
from utils import MF_DATA, CausE_DATA, evaluate_model, evaluate_IPS_model

from naie.datasets import get_data_reference
from naie.context import Context
import moxing as mox


seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)


def extract_data():
    # 将数据集下载到镜像本地，对应的本地目录分别是：
    # /cache/datasets/DatasetService/infer_recommendation_train
    # /cache/datasets/DatasetService/infer_valid
    # /cache/datasets/DatasetService/infer_test
    data_reference1 = get_data_reference(dataset="DatasetService", dataset_entity="infer_recommendation_train", enable_local_cache=True)
    data_reference2 = get_data_reference(dataset="DatasetService", dataset_entity="infer_valid", enable_local_cache=True)
    data_reference3 = get_data_reference(dataset="DatasetService", dataset_entity="infer_test", enable_local_cache=True)
 
    bigtag = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/bigtag.txt',dtype=int)
    choicetag = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/choicetag.txt',dtype=int)
    movie_data = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/movie.txt',dtype=int)
    movie = []
    for i in range(movie_data.shape[0]):
        tmp = movie_data[i,1:]
        movie.append(tmp)

    tag_num = np.max(movie)

    mat = np.zeros((1000,tag_num+1))
    all_data_array = []
    bigtag_array = []
    choicetag_array = []

    # extract deterministic data from bigtag
    for i in range(bigtag.shape[0]):
        if bigtag[i][2] != -1:
            mat[bigtag[i][0]][bigtag[i][2]] = 1
            all_data_array.append([bigtag[i][0],bigtag[i][2],1])
            bigtag_array.append([bigtag[i][0],bigtag[i][2],1])
        if bigtag[i][2] == -1:
            for tag in movie[bigtag[i][1]]:
                mat[bigtag[i][0]][tag] = -1
                all_data_array.append([bigtag[i][0],tag,0])
                bigtag_array.append([bigtag[i][0],tag,0])

    # extract deterministic data from choicetag
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            mat[choicetag[i][0]][choicetag[i][2]] = 1
            all_data_array.append([choicetag[i][0],choicetag[i][2],1])
            choicetag_array.append([choicetag[i][0],choicetag[i][2],1])
        if choicetag[i][2] == -1:
            for tag in movie[choicetag[i][1]]:
                mat[choicetag[i][0]][tag] = -1
                all_data_array.append([choicetag[i][0],tag,0])
                choicetag_array.append([choicetag[i][0],tag,0])
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            for tag in movie[choicetag[i][1]]:
                if mat[choicetag[i][0]][tag] == 0:
                    mat[choicetag[i][0]][tag] = -1
                    all_data_array.append([choicetag[i][0],tag,0])
                    choicetag_array.append([choicetag[i][0],tag,0])

    # Unique
    all_data_array = np.array(all_data_array)
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))
    all_data_array = [tuple(row) for row in all_data_array]
    all_data_array = np.unique(all_data_array, axis=0)
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))

    # Unique
    bigtag_array = np.array(bigtag_array)
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))
    bigtag_array = [tuple(row) for row in bigtag_array]
    bigtag_array = np.unique(bigtag_array, axis=0)
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))

    # Unique
    choicetag_array = np.array(choicetag_array)
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))
    choicetag_array = [tuple(row) for row in choicetag_array]
    choicetag_array = np.unique(choicetag_array, axis=0)
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))

    np.savetxt("/cache/extract_bigtag.txt",np.array(bigtag_array),fmt="%d")
    np.savetxt("/cache/extract_choicetag.txt",np.array(choicetag_array),fmt="%d")
    np.savetxt("/cache/extract_alldata.txt",np.array(all_data_array),fmt="%d")


# train for CausE
def train_CausE(propensity_score_c, propensity_score_t):
    train_data = CausE_DATA(opt.s_c_data, opt.s_t_data)
    val_data = MF_DATA(opt.cause_val_data)
    train_dataloader_s_c = DataLoader(train_data.s_c,
                                      opt.batch_size,
                                      shuffle=True)
    train_dataloader_s_t = DataLoader(train_data.s_t,
                                      opt.batch_size,
                                      shuffle=True)

    pop_item = np.loadtxt('./data/train/pop_item.txt')
    pop_item_val = np.loadtxt('./data/valid/pop_item_val.txt')
    pop_item_tst = np.loadtxt('./data/test/pop_item_tst2.txt')

    inverse_propensity_c = np.reciprocal(propensity_score_c)  # 求倒数
    inverse_propensity_t = np.reciprocal(propensity_score_t)  # 求倒数

    model = getattr(models,
                    opt.model)(train_data.users_num, train_data.items_num,
                               opt.embedding_size, inverse_propensity_c, inverse_propensity_t,
                               opt.reg_uc, opt.reg_ut, opt.reg_utc,
                               opt.reg_ic, opt.reg_it, opt.reg_itc,
                               train_data.s_c[:, :2].tolist(),
                               train_data.s_t[:, :2].tolist(),
                               pop_item, pop_item_val, pop_item_tst)

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader_s_c)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % opt.verbose == 0:
            print('Epoch %d :' % (epoch))
            print('s_c Loss = ', loss.item())

        for i, data in tqdm(enumerate(train_dataloader_s_t)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (mae, mse, rmse, auc) = evaluate_IPS_model(model, val_data, inverse_propensity_c, opt, 1)

        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('s_t Loss = ', loss.item())
            print(
                'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                % (mae, mse, rmse, auc, time() - t1))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
          (best_iter, best_mae, best_mse, best_auc))

    best_model = getattr(models,
                         opt.model)(train_data.users_num, train_data.items_num,
                                    opt.embedding_size, inverse_propensity_c, inverse_propensity_t,
                                    opt.reg_uc, opt.reg_ut, opt.reg_utc,
                                    opt.reg_ic, opt.reg_it, opt.reg_itc,
                                    train_data.s_c[:, :2].tolist(),
                                    train_data.s_t[:, :2].tolist(),
                                    pop_item, pop_item_val, pop_item_tst)
    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))

    print("\n========================= best model =========================")
    mae, mse, rmse, auc = evaluate_IPS_model(best_model, train_data, propensity_score_c, opt, 0)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_IPS_model(best_model, val_data, propensity_score_c, opt, 1)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("===============================================================\n")

    return best_model


def generate_submit(model):
    test_data = np.loadtxt(opt.test_data, dtype=int)
    user = torch.LongTensor(test_data[:, 0]).to(opt.device)
    item = torch.LongTensor(test_data[:, 1]).to(opt.device)
    pred = model.predict(user, item, 2).to(opt.device)
    pred = pred.detach().cpu().numpy()
    # normalize
    pred = (pred-np.min(pred))/(np.max(pred) - np.min(pred))
    pred = pred.reshape(-1,1) 

    train_data = np.loadtxt(opt.train_data, dtype=int)
    for i in range(train_data.shape[0]):
        for j in range(test_data.shape[0]):
            if train_data[i, 0] == test_data[j, 0] and train_data[i, 1] == test_data[j, 1]:
                pred[j] = train_data[i, 2]
                
    valid_data = np.loadtxt(opt.val_all_data, dtype=int)
    for i in range(valid_data.shape[0]):
        for j in range(test_data.shape[0]):
            if valid_data[i, 0] == test_data[j, 0] and valid_data[i, 1] == test_data[j, 1]:
                pred[j] = valid_data[i, 2]

    submit = np.hstack((test_data, pred))
    np.savetxt("/cache/submit.csv", submit, fmt = ('%d','%d','%f'))
 
    # 将结果保存到output目录，最后在比赛界面提交的时候选择对应的训练任务就可以
    mox.file.copy('/cache/submit.csv', os.path.join(Context.get_output_path(), 'submit.csv'))


if __name__ == '__main__':   
    Context.set("model", "CausE")
    Context.set("batch_size", "512")
    Context.set("epoch", "70")
    Context.set("lr", "0.001")
    Context.set("metric", "auc")

    opt.model = Context.get("model")
    opt.batch_size = int(Context.get("batch_size"))
    opt.max_epoch = int(Context.get("epoch"))
    opt.lr = float(Context.get("lr"))
    opt.metric = Context.get("metric")

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    extract_data()

    # opt.model == 'CausE':
    propensity_score_c, propensity_score_t = cal_propensity_score()
    propensity_score_c = np.array(propensity_score_c).astype(float)
    propensity_score_t = np.array(propensity_score_t).astype(float)
    best_model = train_CausE(propensity_score_c, propensity_score_t)
    generate_submit(best_model)

    print('end')
