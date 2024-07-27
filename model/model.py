import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from net.EPResGcn import *
from graph_cteate.graph_generation import *

criterion = nn.MSELoss()
L1_mean = nn.L1Loss(reduction='mean')


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def loss_class(output, target):
    u_loss = criterion(output[:, 0], target[:, 0])
    p_loss = criterion(output[:, 1], target[:, 1])
    loss = criterion(output, target)

    return u_loss, p_loss, loss


class LossFun(nn.Module):
    def __init__(self, device):
        super(LossFun, self).__init__()
        # self.log_var_a = torch.zeros((1,), requires_grad=True)
        # self.log_var_b = torch.zeros((1,), requires_grad=True)

    def forward(self, output, target, index, device):
        index = torch.cat((torch.where(index[:, 0] == 1)[0], torch.where(index[:, 1] == 1)[0]))
        alpha = torch.ones((output.size(0), 1)).to(device)
        alpha[index, 0] = 100
        u_loss = torch.mean(alpha * (0.5 * ((output[:, 0] - target[:, 0]) ** 2) +
                                     0.5 * (torch.abs(output[:, 0] - target[:, 0]))))
        p_loss = torch.mean(alpha * (0.5 * ((output[:, 1] - target[:, 1]) ** 2) +
                                     0.5 * torch.abs(output[:, 1] - target[:, 1])))
        # loss = criterion(output, target)
        return u_loss, p_loss


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def R2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)


def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    r_squared = R2(pred, true)

    return mse, mae, r_squared

def cul_percentage(pred, true):
    pre_data_u = pred[:, 0]
    pre_data_p = pred[:, 1]
    true_data_u = true[:, 0]
    true_data_p = true[:, 1]
    # percentage_error_u = np.mean(np.abs(pre_data_u - true_data_u) / np.abs(true_data_u)) * 100
    # percentage_error_p = np.mean(np.abs(pre_data_p - true_data_p) / np.abs(true_data_p)) * 100

    percentage_error_u = \
        np.mean(np.abs(pre_data_u - true_data_u) / ((np.abs(true_data_u) + np.abs(pre_data_u)) / 2)) * 100
    percentage_error_p = \
        np.mean(np.abs(pre_data_p - true_data_p) / ((np.abs(true_data_p) + np.abs(pre_data_p)) / 2)) * 100

    percentage_error_total = (percentage_error_u + percentage_error_p) / 2

    return percentage_error_u, percentage_error_p, percentage_error_total


def add_gaussian_noise(tensor, device, mean=0, std=0.01):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise.to(device)
    return noisy_tensor


class DlCfd(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.net = self._build_model().to(self.device)
        self.loss_fun = LossFun(self.device)
        self.train_loader, self.test_loader = get_train_test_dataloader(args)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = DnnCfD(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

        return model

    def compute_loss(self, pre, target):
        total_loss = criterion(pre, target)
        return total_loss

    def test(self, epoch):
        self.net.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, points_data in enumerate(self.test_loader):
                point_data = points_data.to(self.device)
                points = point_data.x
                p_type = point_data.p_type
                points = torch.cat((points, p_type), dim=1)

                conditions = point_data.boundary_mask
                rcr = point_data.rcr
                conditions = torch.cat((conditions, rcr), dim=1)

                index = point_data.edge_index

                attr = point_data.edge_attr
                e_type = point_data.e_type
                attr = torch.cat((attr, e_type), dim=1)

                batch = point_data.batch
                time_condition = point_data.time
                pre_data = self.net(points, conditions, index, attr, batch, time_condition)
                true = point_data.y.detach().cpu().numpy()
                pre_data = pre_data.detach().cpu().numpy()
                preds.append(pre_data)
                trues.append(true)
        preds = np.concatenate([arr for arr in preds], axis=0)
        trues = np.concatenate([arr for arr in trues], axis=0)
        mse, mae, r_squared = metric(preds, trues)
        per_u, per_p, per_all = cul_percentage(preds, trues)
        return mse, mae, r_squared, per_u, per_p, per_all

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        auto_loss = AutomaticWeightedLoss(2).to(self.device)

        milestones = [10, 20, 30, 40, 60, 80]
        # model_optim = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        model_optim = optim.Adam([{'params': self.net.parameters()}, {'params': auto_loss.parameters()}],
                                 lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones, gamma=0.5)
        for epoch in range(self.args.train_epochs):
            preds = []
            trues = []
            self.net.train()  # train model
            for i, point_data in enumerate(self.train_loader):
                point_data = point_data.to(self.device)
                points = point_data.x
                p_type = point_data.p_type
                points = torch.cat((points, p_type), dim=1)

                conditions = point_data.boundary_mask

                rcr = point_data.rcr
                conditions = torch.cat((conditions, rcr), dim=1)

                index = point_data.edge_index

                attr = point_data.edge_attr
                e_type = point_data.e_type
                attr = torch.cat((attr, e_type), dim=1)

                batch = point_data.batch
                time_condition = point_data.time
                pre_data = self.net(points, conditions, index, attr, batch, time_condition)
                u_loss, p_loss, total_loss = loss_class(pre_data, point_data.y)
                total_loss = auto_loss(u_loss, p_loss)

                pred = pre_data.detach().cpu().numpy()
                true = point_data.y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                model_optim.zero_grad()
                total_loss.backward()
                model_optim.step()

            scheduler.step()
            mse, mae, r_squared, per_u, per_p, per_all = self.test(epoch)
            print(f'epoch:{epoch}, mse:{mse}, mae:{mae}, r_2:{r_squared}, per_u:{per_u}, per_p:{per_p}, per_all:{per_all}')

        best_model_path = path + '/' + 'model.pth'
        torch.save(self.net.state_dict(), best_model_path)

        return self.net


