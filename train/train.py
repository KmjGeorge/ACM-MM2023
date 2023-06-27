import datetime
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
from sklearn import metrics
from configs.nsconfig import dataconfig, audioconfig, trainconfig, cavmaeconfig
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


def save(info, model, savename, epoch, start_epoch):
    logs = pd.DataFrame({'loss': info.loss_list,
                         'map': info.map_list,
                         'uar': info.uar_list,
                         'val_loss': info.val_loss_list,
                         'val_map': info.val_map_list,
                         'val_uar': info.val_uar_list,
                         'lr': info.lr_list
                         })
    logs.to_csv('../logs/{}_logs.csv'.format(savename), index=True)
    torch.save(model.state_dict(),
               "../weights/{}_epoch{}.pt".format(savename, start_epoch + epoch))  # 每轮保存一次参数


def train(audio_model, train_loader, val_loader, start_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    epoch = 0
    info = TrainingInfo()

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    # mlp_list = ['mlp_head1.0.weight', 'mlp_head1.0.bias', 'mlp_head1.1.weight', 'mlp_head1.1.bias',
    #             'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
    #             'mlp_head3.0.weight', 'mlp_head3.0.bias', 'mlp_head3.1.weight', 'mlp_head3.1.bias',
    #             'mlp_head4.0.weight', 'mlp_head4.0.bias', 'mlp_head4.1.weight', 'mlp_head4.1.bias',
    #             'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
    #             'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
    #             'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight',
    #             'mlp_head_concat.1.bias']
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
                'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight',
                'mlp_head_concat.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if trainconfig['freeze_base'] == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(trainconfig['head_lr']))
    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': trainconfig['lr']},
         {'params': mlp_params, 'lr': trainconfig['lr'] * trainconfig['head_lr']}],
        weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [trainconfig['lr'], mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in base_params) / 1e6))

    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if trainconfig['lr_adapt'] == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                               patience=trainconfig['lr_patience'], verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(
            range(trainconfig['lrscheduler_start'], 1000, trainconfig['lrscheduler_step'])),
                                                         gamma=trainconfig['lrscheduler_decay'])
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(
            trainconfig['lrscheduler_start'], trainconfig['lrscheduler_decay'], trainconfig['lrscheduler_step']))
    loss_fn = trainconfig['loss']

    epoch += 1
    scaler = GradScaler()
    print("start training...")
    audio_model.train()
    while epoch < trainconfig['n_epochs'] + 1:
        audio_model.train()

        TP = [0, 0, 0, 0]
        FN = [0, 0, 0, 0]
        FP = [0, 0, 0, 0]
        Recall = [0., 0., 0., 0.]
        AP = [0., 0., 0., 0.]
        Total_avg_loss = 0.
        batch_sum_loss = 0.
        iter = 0

        loop = tqdm(train_loader)
        for a_input, v, labels, _ in loop:
            iter += 1
            # 随机从10帧中采样1帧
            index = np.random.randint(0, 10)
            v_input = v[:, :, index, :, :]
            v_input = v_input.float()
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.float().to(device)
            # 计算loss和uar
            with autocast():
                audio_output = audio_model(a_input, v_input, cavmaeconfig['ftmode']).to(device)  # (batch, 4)
                loss = loss_fn(audio_output, labels)

            for i, (logit, y_true) in enumerate(zip(torch.sigmoid(audio_output).T, labels.T)):  # (4, batch)
                y_pred = torch.round(logit)  # 超过0.5为1 否则为0
                for j in range(len(y_true)):
                    if (y_pred[j].item() == 1.) and (y_true[j].item() == 1.):
                        TP[i] += 1
                    elif (y_pred[j].item() == 0.) and (y_true[j].item() == 1.):
                        FN[i] += 1
                    elif (y_pred[j].item() == 1.) and (y_true[j].item() == 0.):
                        FP[i] += 1
                if TP[i] + FN[i] != 0:
                    Recall[i] = TP[i] / (TP[i] + FN[i])
                else:
                    Recall[i] = 0
                if TP[i] + FP[i] != 0:
                    AP[i] = TP[i] / (TP[i] + FP[i])
                else:
                    AP[i] = 0
            mAP = sum(AP) / 4
            UAR = sum(Recall) / 4
            batch_sum_loss += loss.item()
            Total_avg_loss = batch_sum_loss / iter
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description('Epoch{}'.format(epoch))
            loop.set_postfix(Recall=[round(re, 3) for re in Recall], AP=[round(ap, 3) for ap in AP], mAP=mAP, UAR=UAR, loss=Total_avg_loss,
                             lr=optimizer.param_groups[0]['lr'], lr_head=optimizer.param_groups[1]['lr'])
        # 每轮验证一次
        val_UAR, val_mAP, val_loss = validate(audio_model, val_loader)

        if trainconfig['save_model'] == True:
            info.uar_list.append(sum(AP) / 4)
            info.map_list.append(sum(Recall) / 4)
            info.loss_list.append(Total_avg_loss)
            info.val_uar_list.append(val_UAR)
            info.val_map_list.append(val_mAP)
            info.val_loss_list.append(val_loss)
            info.lr_list.append(optimizer.param_groups[0]['lr'])
            save(info, audio_model, epoch=epoch, savename=trainconfig['savename'], start_epoch=start_epoch)

        scheduler.step()
        epoch += 1


def validate(audio_model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    TP = [0, 0, 0, 0]
    FP = [0, 0, 0, 0]
    FN = [0, 0, 0, 0]
    Recall = [0., 0., 0., 0.]
    AP = [0., 0., 0., 0.]

    Total_Loss_avg = 0.
    batch_sum_loss = 0.
    loss_fn = trainconfig['loss']
    iter = 0
    loop = tqdm(val_loader)
    with torch.no_grad():
        for a_input, v, labels, _ in loop:
            iter += 1
            a_input = a_input.to(device)
            labels = labels.float().to(device)
            audio_output = torch.zeros(size=(a_input.size(0), 4)).to(device)
            # 10帧输入的输出取平均
            for index in range(10):
                v_input = v[:, :, index, :, :]
                v_input = v_input.float().to(device)
                with autocast():
                    audio_output += audio_model(a_input, v_input, cavmaeconfig['ftmode'])
            audio_output /= 10

            # 计算loss和uar
            with autocast():
                audio_output = audio_model(a_input, v_input, cavmaeconfig['ftmode']).to(device)
                loss = loss_fn(audio_output, labels)

            for i, (logit, y_true) in enumerate(zip(torch.sigmoid(audio_output).T, labels.T)):
                y_pred = torch.round(logit)  # 超过0.5为1 否则为0
                for j in range(len(y_true)):
                    if (y_pred[j].item() == 1.) and (y_true[j].item() == 1.):
                        TP[i] += 1
                    elif (y_pred[j].item() == 0.) and (y_true[j].item() == 1.):
                        FN[i] += 1
                    elif (y_pred[j].item() == 1.) and (y_true[j].item() == 0.):
                        FP[i] += 1
                if TP[i] + FN[i] != 0:
                    Recall[i] = TP[i] / (TP[i] + FN[i])
                else:
                    Recall[i] = 0
                if TP[i] + FP[i] != 0:
                    AP[i] = TP[i] / (TP[i] + FP[i])
                else:
                    AP[i] = 0
            mAP = sum(AP) / 4
            UAR = sum(Recall) / 4
            batch_sum_loss += loss.item()
            Total_avg_loss = batch_sum_loss / iter
            loop.set_description('Validation')
            loop.set_postfix(Recall=[round(re, 3) for re in Recall], AP=[round(ap, 3) for ap in AP], mAP=mAP, UAR=UAR,
                             loss=Total_avg_loss)

    audio_model.train()
    return UAR, mAP, Total_Loss_avg


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000  # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
