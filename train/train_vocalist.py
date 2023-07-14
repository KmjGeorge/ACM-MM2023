import torch
from torch import nn
import numpy as np

from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import recall_score, average_precision_score, precision_score
from configs.nsconfig import dataconfig, audioconfig, trainconfig, cavmaeconfig, vocalistconfig
from tqdm import tqdm
import pandas as pd


class TrainingInfo:
    def __init__(self):
        self.loss_list = []
        self.map_list = []
        self.uar_list = []
        self.val_loss_list = []
        self.val_map_list = []
        self.val_uar_list = []
        self.lr_list = []
        self.lr_head_list = []


def save(info, model, savename, epoch, start_epoch):
    logs = pd.DataFrame({'loss': info.loss_list,
                         'map': info.map_list,
                         'uar': info.uar_list,
                         'val_loss': info.val_loss_list,
                         'val_map': info.val_map_list,
                         'val_uar': info.val_uar_list,
                         'lr': info.lr_list,
                         })
    logs.to_csv('../logs/{}_logs.csv'.format(savename), index=True)
    torch.save(model.state_dict(),
               "../weights/{}_epoch{}.pt".format(savename, start_epoch + epoch))  # 每轮保存一次参数


def train_vocalist(model, train_loader, val_loader, start_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    epoch = 0
    info = TrainingInfo()

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=trainconfig['lr'],
                                 weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(
        range(trainconfig['lrscheduler_start'], 1000, trainconfig['lrscheduler_step'])),
                                                     gamma=trainconfig['lrscheduler_decay'])

    loss_fn = trainconfig['loss']

    epoch += 1
    scaler = GradScaler()
    print("start training...")
    model.train()
    while epoch < trainconfig['n_epochs'] + 1:
        train_uar, train_mAP, train_loss = train_vocalist_per_epoch(model, train_loader, device, loss_fn, optimizer,
                                                                    scaler, epoch)
        if epoch % trainconfig['validate_step'] == 0:
            val_uar, val_mAP, val_loss = validate_vocalist_per_epoch(model, val_loader)
            info.val_uar_list.append(val_uar)
            info.val_map_list.append(val_mAP)
            info.val_loss_list.append(val_loss)
        else:
            info.val_uar_list.append(0)
            info.val_map_list.append(0)
            info.val_loss_list.append(0)
        info.uar_list.append(train_uar)
        info.map_list.append(train_mAP)
        info.loss_list.append(train_loss)
        info.lr_list.append(optimizer.param_groups[0]['lr'])

        if trainconfig['save_model'] == True:
            save(info, model, epoch=epoch, savename=trainconfig['savename'], start_epoch=start_epoch)

        scheduler.step()
        epoch += 1


def train_vocalist_per_epoch(model, train_loader, device, loss_fn, optimizer, scaler, epoch):
    model.train()

    Recall = [0., 0., 0., 0.]
    AP = [0., 0., 0., 0.]
    all_labels = [np.array([]), np.array([]), np.array([]), np.array([])]
    all_preds = [np.array([]), np.array([]), np.array([]), np.array([])]
    Total_avg_loss = 0.
    batch_sum_loss = 0.
    iter = 0

    loop = tqdm(train_loader)
    for a_input, v_input, labels, _ in loop:
        iter += 1
        v_shape = v_input.shape  # (B, 4, C, D, H, W)
        v_input = v_input.reshape(v_shape[0], v_shape[1], v_shape[2] * v_shape[3], v_shape[4], v_shape[5])
        a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        # 计算loss和uar
        with autocast():
            output = model(v_input, a_input, vocalistconfig['pool']).to(device)  # (batch, 4)
            loss = loss_fn(output, labels)
            # loss = loss_fn(output[:, 0], labels[:, 0])
            # loss += loss_fn(output[:, 1], labels[:, 1])
            # loss += loss_fn(output[:, 2], labels[:, 2])
            # loss += loss_fn(output[:, 3], labels[:, 3])
        with torch.no_grad():
            for i in range(4):  # 提取对每个说话人的预测结果
                # y_pred = torch.round(torch.sigmoid(output[:, i])).detach().to('cpu').numpy()
                y_pred = torch.sigmoid(output[:, i]).detach().to('cpu').numpy()  # (batch, )
                for b_index in range(a_input.size(0)):
                    if y_pred[b_index] > trainconfig['cls_threshold']:
                        y_pred[b_index] = 1.
                    else:
                        y_pred[b_index] = 0.
                y_true = labels[:, i].to('cpu').numpy()
                all_preds[i] = np.concatenate((all_preds[i], y_pred))
                all_labels[i] = np.concatenate((all_labels[i], y_true))
                Recall[i] = recall_score(all_labels[i], all_preds[i], zero_division=0.,  average='macro')
                AP[i] = precision_score(all_labels[i], all_preds[i], zero_division=0.,  average='macro')
            mAP = sum(AP) / 4
            UAR = sum(Recall) / 4

            batch_sum_loss += loss.item()
            Total_avg_loss = batch_sum_loss / iter

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description('Epoch{}'.format(epoch))
        loop.set_postfix(Recall=[round(re, 3) for re in Recall],
                         AP=[round(ap, 3) for ap in AP],
                         mAP=mAP,
                         UAR=UAR,
                         loss=Total_avg_loss,
                         lr=optimizer.param_groups[0]['lr'])
    return UAR, mAP, Total_avg_loss


def validate_vocalist_per_epoch(model, val_loader):
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_labels = [np.array([]), np.array([]), np.array([]), np.array([])]
    all_preds = [np.array([]), np.array([]), np.array([]), np.array([])]
    Recall = [0., 0., 0., 0.]
    AP = [0., 0., 0., 0.]

    batch_sum_loss = 0.
    loss_fn = trainconfig['loss']
    iter = 0
    loop = tqdm(val_loader)
    with torch.no_grad():
        for a_input, v_input, labels, _ in loop:
            iter += 1
            labels = labels.float().to(device)
            v_shape = v_input.shape  # (B, 4, C, D, H, W)
            v_input = v_input.reshape(v_shape[0], v_shape[1], v_shape[2] * v_shape[3], v_shape[4], v_shape[5])
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.float().to(device, non_blocking=True)
            # 计算loss和uar
            with autocast():
                output = model(v_input, a_input, vocalistconfig['pool']).to(device)  # (batch, 4)
                # print(output)
                loss = loss_fn(output, labels)
                # loss = loss_fn(output[:, 0], labels[:, 0])
                # loss += loss_fn(output[:, 1], labels[:, 1])
                # loss += loss_fn(output[:, 2], labels[:, 2])
                # loss += loss_fn(output[:, 3], labels[:, 3])

            for i in range(4):  # 提取对每个说话人的预测结果
                # y_pred = torch.round(torch.sigmoid(output[:, i])).detach().to('cpu').numpy()
                y_pred = torch.sigmoid(output[:, i]).detach().to('cpu').numpy()  # (batch, )
                for b_index in range(a_input.size(0)):
                    if y_pred[b_index] > trainconfig['cls_threshold']:
                        y_pred[b_index] = 1.
                    else:
                        y_pred[b_index] = 0.

                y_true = labels[:, i].to('cpu').numpy()
                all_preds[i] = np.concatenate((all_preds[i], y_pred))
                all_labels[i] = np.concatenate((all_labels[i], y_true))
                Recall[i] = recall_score(all_labels[i], all_preds[i], zero_division=0., average='macro')
                AP[i] = precision_score(all_labels[i], all_preds[i], zero_division=0., average='macro')
            mAP = sum(AP) / 4
            UAR = sum(Recall) / 4
            batch_sum_loss += loss.item()
            Total_avg_loss = batch_sum_loss / iter
            loop.set_description('Validation')
            loop.set_postfix(Recall=[round(re, 3) for re in Recall],
                             AP=[round(ap, 3) for ap in AP],
                             mAP=mAP,
                             UAR=UAR,
                             loss=Total_avg_loss,
                             )
    model.train()
    return UAR, mAP, Total_avg_loss
