from config import opt
from propensity_score import cal_propensity_score
import os
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from metrics import AUC
from utils import MF_DATA, CausE_DATA, evaluate_model, evaluate_IPS_model
import numpy as np
import argparse
import random
import torch
import copy


seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)


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
                    opt.model)(train_data.users_num, train_data.items_num, opt.embedding_size,
                               inverse_propensity_c, inverse_propensity_t,
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

        (mae, mse, rmse, auc) = evaluate_IPS_model(model, val_data, inverse_propensity_c, opt, is_val=1)

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
                         opt.model)(train_data.users_num, train_data.items_num, opt.embedding_size,
                                    inverse_propensity_c, inverse_propensity_t,
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
    mae, mse, rmse, auc = evaluate_IPS_model(best_model, train_data, inverse_propensity_c, opt, is_val=0)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_IPS_model(best_model, val_data, inverse_propensity_c, opt, is_val=1)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("==============================================================\n")

    return best_model


# gengerate submit file
def generate_submit(model):
    test_data = np.loadtxt(opt.test_data, dtype=int)
    user = torch.LongTensor(test_data[:, 0]).to(opt.device)
    item = torch.LongTensor(test_data[:, 1]).to(opt.device)
    pred = model.predict(user, item, is_val=2).to(opt.device)
    pred = pred.detach().cpu().numpy()
    # normalize
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = pred.reshape(-1, 1)

    # 可以优化：存成文件
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
    np.savetxt("submit.csv", submit, fmt=('%d', '%d', '%f'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model', default='CausE')  # 换模型的话，改这里的参数
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--metric',
                        default='auc',
                        choices=["mae", "mse", "auc"])

    args = parser.parse_args()
    opt.model = args.model
    opt.batch_size = args.batch_size
    opt.max_epoch = args.epoch
    opt.lr = args.lr
    opt.metric = args.metric

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))
    print('model is', opt.model)

    # opt.model == 'CausE':
    propensity_score_c, propensity_score_t = cal_propensity_score()
    propensity_score_c = np.array(propensity_score_c).astype(float)
    propensity_score_t = np.array(propensity_score_t).astype(float)
    best_model = train_CausE(propensity_score_c, propensity_score_t)
    generate_submit(best_model)

    print('end')
