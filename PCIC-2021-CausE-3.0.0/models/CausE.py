import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from .loss import IPSLoss
from config import opt


class CausE(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, inverse_propensity_c, inverse_propensity_t,
                 reg_uc, reg_ut, reg_utc, reg_ic, reg_it, reg_itc,
                 s_c, s_t, pop_item, pop_item_val, pop_item_tst, device='cpu'):
        super(CausE, self).__init__()
        self.inverse_propensity_c = inverse_propensity_c
        self.inverse_propensity_t = inverse_propensity_t

        self.user_e = nn.Embedding(num_users, embedding_size)
        self.item_e_c = nn.Embedding(num_items, embedding_size)
        self.item_e_t = nn.Embedding(num_items, embedding_size)
        self.user_b = nn.Embedding(num_users, 1)
        self.item_b = nn.Embedding(num_items, 1)

        self.reg_uc = reg_uc
        self.reg_ut = reg_ut
        self.reg_utc = reg_utc
        self.reg_ic = reg_ic
        self.reg_it = reg_it
        self.reg_itc = reg_itc
        self.s_c = s_c
        self.s_t = s_t
        self.pop_item = pop_item
        self.pop_item_val = pop_item_val
        self.pop_item_tst = pop_item_tst

        self.apply(self._init_weights)

        # self.loss_c = nn.MSELoss()
        # self.loss_t = nn.MSELoss()
        self.loss_c = IPSLoss(device)
        self.loss_t = IPSLoss(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)


    def forward(self, user, item, is_val=0):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e_c(item)  #
        item0 = item.cpu().numpy()
        item_list_ = item.cpu().numpy()

        preds = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        # 对传统打分的值做一个激活，使它变成正向的
        for i in range(len(preds)):
            if preds[i] <= 0:
                preds[i] = torch.exp(preds[i])
            else:
                preds[i] = preds[i] + 1
        if is_val == 0:
            weight = torch.Tensor(
                list(map(lambda x: self.pop_item[item_list_[x]],
                        range(0, len(item0))))).to(opt.device)
        elif is_val == 1:
            weight = torch.Tensor(
                list(map(lambda x: self.pop_item[item_list_[x]] + opt.pop_drift * np.abs(
                    self.pop_item_val[item_list_[x]] - self.pop_item[item_list_[x]]),
                        range(0, len(item0))))).to(opt.device)
        else:
            weight = torch.Tensor(
                list(map(lambda x: self.pop_item[item_list_[x]] + opt.pop_drift * np.abs(
                    self.pop_item_tst[item_list_[x]] - self.pop_item[item_list_[x]]),
                         range(0, len(item0))))).to(opt.device)
        weight = np.power(weight, opt.pop_exp)
        preds = preds * weight.unsqueeze(dim=1)
        preds = torch.log(torch.sigmoid(preds))
        preds += self.user_b(user) + self.item_b(item)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list, control):
        user_embedding = self.user_e(user_list)
        item_embedding_c = self.item_e_c(item_list)
        # item0 = item_list.cpu().numpy()
        # item_list_ = item_list.cpu().numpy()

        if control == True:
            dot_c = (user_embedding * item_embedding_c).sum(dim=1, keepdim=True)
            for i in range(len(dot_c)):
                if dot_c[i] <= 0:
                    dot_c[i] = torch.exp(dot_c[i])
                else:
                    dot_c[i] = dot_c[i] + 1
            # weight = torch.Tensor(list(map(lambda x: (self.pop_item[item_list_[x]]),
            #                         range(0, len(item0))))).to(opt.device)
            # weight = np.power(weight, opt.pop_exp)
            # dot_c_pop = dot_c * weight.unsqueeze(dim=1)
            dot_c_pop = torch.log(torch.sigmoid(dot_c))
            pred_c = dot_c_pop + self.user_b(user_list) + self.item_b(item_list)
            pred_c = pred_c.squeeze()

            # loss = self.loss_c(pred_c, label_list)
            loss = self.loss_c(pred_c, label_list, self.inverse_propensity_c, item_list)
            loss_reg_c = self.reg_uc * torch.norm(user_embedding, 2)  # 改
            loss_reg_c += self.reg_ic * torch.norm(item_embedding_c, 2)  # 改
            loss_reg_t = 0
            loss_reg_tc = 0

        else:
            user_embedding = self.user_e(user_list)
            item_embedding_t = self.item_e_t(item_list)  #
            dot_t = (user_embedding * item_embedding_t).sum(dim=1, keepdim=True)
            # for i in range(len(dot_t)):
            #     if dot_t[i] <= 0:
            #         dot_t[i] = torch.exp(dot_t[i])
            #     else:
            #         dot_t[i] = dot_t[i] + 1
            # dot_t_pop = torch.log(torch.sigmoid(dot_t))
            pred_t = dot_t + self.user_b(user_list) + self.item_b(item_list)
            pred_t = pred_t.squeeze()

            # loss = self.loss_t(pred_t, label_list)
            loss = self.loss_t(pred_t, label_list, self.inverse_propensity_t, item_list)

            loss_reg_c = self.reg_uc * torch.norm(user_embedding, 2)  # 改
            loss_reg_c += self.reg_ic * torch.norm(item_embedding_c, 2)  # 改
            loss_reg_t = self.reg_ut * torch.norm(user_embedding, 1)  # 改为一范数
            loss_reg_t += self.reg_it * torch.norm(item_embedding_t, 1)  # 改为一范数

            # 结合了因果推断的信息
            # loss_reg_tc = self.reg_itc * torch.norm(user_embedding_c - user_embedding_t, 2)  # 改
            loss_reg_tc = self.reg_itc * torch.norm(item_embedding_c - item_embedding_t, 2)  # 改

        return loss + loss_reg_c + loss_reg_t + loss_reg_tc

    def predict(self, user, item, is_val):
        return self.forward(user, item, is_val)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
