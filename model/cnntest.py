import torch.nn as nn
import timm
import torchvision
import torch
from sklearn.metrics import recall_score, average_precision_score, precision_score
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from train.train import TrainingInfo
from configs.nsconfig import *
from tqdm import tqdm
import pandas as pd

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

def get_resnet50(pretrained=True):
    net = torchvision.models.resnet50(pretrained=False)
    if pretrained:
        net.load_state_dict(torch.load('../weights/resnet50-0676ba61.pth'))
    del net.fc
    net.add_module('fc', nn.Linear(2048, 4))
    return net


def train_resnet50(model, train_loader, val_loader, start_epoch):
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
        train_uar, train_mAP, train_loss = train_resnet50_per_epoch(model, train_loader, device, loss_fn, optimizer,
                                                                     scaler, epoch)
        val_uar, val_mAP, val_loss = validate_resnet50_per_epoch(model, val_loader)

        if trainconfig['save_model'] == True:
            info.uar_list.append(train_uar)
            info.map_list.append(train_mAP)
            info.loss_list.append(train_loss)
            info.val_uar_list.append(val_uar)
            info.val_map_list.append(val_mAP)
            info.val_loss_list.append(val_loss)
            info.lr_list.append(optimizer.param_groups[0]['lr'])
            save(info, model, epoch=epoch, savename=trainconfig['savename'], start_epoch=start_epoch)

        scheduler.step()
        epoch += 1


def train_resnet50_per_epoch(model, train_loader, device, loss_fn, optimizer, scaler, epoch):
    model.train()

    # TP = [0, 0, 0, 0]
    # FN = [0, 0, 0, 0]
    # FP = [0, 0, 0, 0]
    Recall = [0., 0., 0., 0.]
    AP = [0., 0., 0., 0.]
    all_labels = [np.array([]), np.array([]), np.array([]), np.array([])]
    all_preds = [np.array([]), np.array([]), np.array([]), np.array([])]
    Total_avg_loss = 0.
    batch_sum_loss = 0.
    iter = 0

    loop = tqdm(train_loader)
    for _, v, labels, _ in loop:
        iter += 1
        # 随机从10帧中采样1帧
        # index = np.random.randint(0, 10)
        index = np.random.randint(0, 10)
        v_input = v[:, :, index, :, :]
        v_input = v_input.float()
        # a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
        labels = labels.float().to(device)
        # 计算loss和uar
        with autocast():
            audio_output = model(v_input).to(device)  # (batch, 4)
            # loss = loss_fn(audio_output, labels)
            loss = loss_fn(audio_output[:, 0], labels[:, 0])
            loss += loss_fn(audio_output[:, 1], labels[:, 1])
            loss += loss_fn(audio_output[:, 2], labels[:, 2])
            loss += loss_fn(audio_output[:, 3], labels[:, 3])
        with torch.no_grad():
            for i in range(4):  # 提取对每个说话人的预测结果
                y_pred = torch.round(torch.sigmoid(audio_output[:, i])).detach().to('cpu').numpy()
                y_true = labels[:, i].to('cpu').numpy()
                all_preds[i] = np.concatenate((all_preds[i], y_pred))
                all_labels[i] = np.concatenate((all_labels[i], y_true))
                Recall[i] = recall_score(all_labels[i], all_preds[i], zero_division=0.)
                AP[i] = precision_score(all_labels[i], all_preds[i], zero_division=0.)
            mAP = sum(AP) / 4
            UAR = sum(Recall) / 4

            batch_sum_loss += loss.item()
            Total_avg_loss = batch_sum_loss / iter

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description('Epoch{}'.format(epoch))
        loop.set_postfix(Recall=[round(re, 3) for re in Recall],
                         AP=[round(ap, 3) for ap in AP],
                         mAP=mAP, UAR=UAR,
                         loss=Total_avg_loss,
                         lr=optimizer.param_groups[0]['lr'])
    return UAR, mAP, Total_avg_loss


def validate_resnet50_per_epoch(model, val_loader):
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_labels = [np.array([]), np.array([]), np.array([]), np.array([])]
    all_preds = [np.array([]), np.array([]), np.array([]), np.array([])]
    Recall = [0., 0., 0., 0.]
    AP = [0., 0., 0., 0.]

    Total_Loss_avg = 0.
    batch_sum_loss = 0.
    loss_fn = trainconfig['loss']
    iter = 0
    loop = tqdm(val_loader)
    with torch.no_grad():
        for _, v, labels, _ in loop:
            iter += 1
            labels = labels.float().to(device)
            index = np.random.randint(0, 10)
            v_input = v[:, :, index, :, :]
            v_input = v_input.float().to(device)

            # 计算loss和uar
            with autocast():
                audio_output = model(v_input).to(device)  # (batch, 4)
                # loss = loss_fn(audio_output, labels)
                loss = loss_fn(audio_output[:, 0], labels[:, 0])
                loss += loss_fn(audio_output[:, 1], labels[:, 1])
                loss += loss_fn(audio_output[:, 2], labels[:, 2])
                loss += loss_fn(audio_output[:, 3], labels[:, 3])
            for i in range(4):  # 提取对每个说话人的预测结果
                y_pred = torch.round(torch.sigmoid(audio_output[:, i])).detach().to('cpu').numpy()
                y_true = labels[:, i].to('cpu').numpy()
                all_preds[i] = np.concatenate((all_preds[i], y_pred))
                all_labels[i] = np.concatenate((all_labels[i], y_true))
                Recall[i] = recall_score(all_labels[i], all_preds[i], zero_division=0.)
                AP[i] = precision_score(all_labels[i], all_preds[i], zero_division=0.)
                mAP = sum(AP) / 4
                UAR = sum(Recall) / 4
                batch_sum_loss += loss.item()
                Total_avg_loss = batch_sum_loss / iter
            loop.set_description('Validation')
            loop.set_postfix(Recall=[round(re, 3) for re in Recall], AP=[round(ap, 3) for ap in AP], mAP=mAP, UAR=UAR,
                             loss=Total_avg_loss)
    model.train()
    return UAR, mAP, Total_Loss_avg