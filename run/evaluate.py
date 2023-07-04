import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from torch.cuda.amp import autocast
from model.cnntest import train_resnet50
from dataset.dataloader import get_testloader, get_dataloader
from torchsummary import summary
from model.cnntest import get_resnet50
from tqdm import tqdm
from configs.nsconfig import *
import random
import matplotlib.pyplot as plt
import pandas as pd


def show_logs(log_path):
    df = pd.read_csv(log_path)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(df['loss'], label='loss')
    ax[0][0].plot(df['val_loss'], label='val_loss')
    ax[0][0].set_title('Loss')


    ax[0][1].plot(df['uar'], label='uar')
    ax[0][1].plot(df['val_uar'], label='val_uar')
    ax[0][1].set_title('UAR')

    ax[1][0].plot(df['map'], label='map')
    ax[1][0].plot(df['val_map'], label='val_map')
    ax[1][0].set_title('MAP')

    ax[1][1].plot(df['lr'], label='lr')
    ax[1][1].plot(df['lr_head'], label='lr_head')
    ax[1][1].set_title('Learning Rate')

    for i in range(2):
        for j in range(2):
            ax[i][j].legend()
            ax[i][j].set_xlabel('Epoch')

    plt.tight_layout()

    plt.savefig(log_path.replace('logs', 'figures').replace('csv', 'jpg'))
    plt.show()



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, test_loader):
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
    loop = tqdm(test_loader)
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
            loop.set_description('Test')
            loop.set_postfix(Recall=[round(re, 3) for re in Recall], AP=[round(ap, 3) for ap in AP], mAP=mAP, UAR=UAR,
                             loss=Total_avg_loss)
    model.train()
    return UAR, mAP, Total_Loss_avg


if __name__ == '__main__':
    '''
    setup_seed(100)
    resnet50 = get_resnet50(pretrained=False)
    weights = torch.load('../weights/resnet50-1e-5 0.95_epoch2.pt')
    new_weights = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '')
        new_weights[new_k] = v
    resnet50.load_state_dict(new_weights)
    testloader = get_testloader('mean')
    # trainloader, valloader = get_dataloader('mean')
    summary(resnet50, input_size=(3, 224, 224), device='cpu')
    evaluate(resnet50, testloader)
    '''
    show_logs('../logs/cavmaeft-all concat4 batch8 1e-5 head1 0.9_logs.csv')
