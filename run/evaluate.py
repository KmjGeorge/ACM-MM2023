import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from torch.cuda.amp import autocast
from dataset.dataloader import get_testloader, get_dataloader
from torchsummary import summary
from model.cnntest import get_resnet50
from tqdm import tqdm
from configs.nsconfig import *
import random
import matplotlib.pyplot as plt
import pandas as pd

from model.vocalist import SyncTransformer


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
    try:
        ax[1][1].plot(df['lr_head'], label='lr_head')
    except:
        pass
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
    all_preds = [np.array([]), np.array([]), np.array([]), np.array([])]
    all_ids = np.array([])
    iter = 0
    loop = tqdm(test_loader)
    with torch.no_grad():
        for a_input, v_input, _, id in loop:
            # print(id)
            # print(a_input.shape, v_input.shape)
            iter += 1
            v_shape = v_input.shape  # (B, 4, C, D, H, W)
            v_input = v_input.reshape(v_shape[0], v_shape[1], v_shape[2] * v_shape[3], v_shape[4], v_shape[5])
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.float().to(device, non_blocking=True)
            # 计算loss和uar
            with autocast():
                output = model(v_input, a_input, vocalistconfig['pool']).to(device)  # (batch, 4)

            for i in range(4):  # 提取对每个说话人的预测结果
                # y_pred = torch.round(torch.sigmoid(output[:, i])).detach().to('cpu').numpy()
                y_pred = torch.sigmoid(output[:, i]).detach().to('cpu').numpy()  # (batch, )
                for b_index in range(a_input.size(0)):
                    if y_pred[b_index] > trainconfig['cls_threshold']:
                        y_pred[b_index] = 1.
                    else:
                        y_pred[b_index] = 0.

                all_preds[i] = np.concatenate((all_preds[i], y_pred))
            all_ids = np.concatenate((all_ids, id))
            loop.set_description('Evaluation')
    model.train()
    return all_preds, all_ids


def getuar(pred_csv, label_csv):
    df1 = pd.read_csv(pred_csv)
    df2 = pd.read_csv(label_csv)
    pred_1, pred_2, pred_3, pred_4 = df1['label_1'], df1['label_2'], df1['label_3'], df1['label_4']
    label_1, label_2, label_3, label_4 = df2['label_1'], df2['label_2'], df2['label_3'], df2['label_4']
    average = 'macro'
    recall1, recall2, recall3, recall4 = recall_score(label_1, pred_1, average=average), recall_score(label_2, pred_2, average=average), recall_score(label_3, pred_3, average=average), recall_score(label_4, pred_4, average=average)
    print('Recall:', recall1, recall2, recall3, recall4)
    print('UAR:', (recall1 + recall2 + recall3 + recall4 )/ 4)

if __name__ == '__main__':
    getuar('../output/val1.csv', 'D:/Datasets/NextSpeaker/next_speaker_val.csv')
    assert False
    setup_seed(100)
    # show_logs('../logs/VocaList 15frame pretrain norm(255) backward1 concat4 batch12 1e-4 5 0.9 threhold0.5 alter_logs.csv')
    # assert False
    model = SyncTransformer()
    weights = torch.load(
        '../weights/VocaList 15frame pretrain norm(255) backward1 concat4 batch12 1e-4 5 0.9 threhold0.5 alter_epoch30.pt')
    new_weights = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '')
        new_weights[new_k] = v
    model.load_state_dict(new_weights)
    train_loader, val_loader = get_dataloader(None, 15, False)
    # test_loader = get_testloader(None, 15, False)
    # trainloader, valloader = get_dataloader('mean')
    # summary(model, input_size=[(4, 15*3, 96, 96), (1, 80, 1103)], device='cpu')
    results, all_ids = evaluate(model, val_loader)
    dict = {'id': all_ids, 'label_1': results[0], 'label_2': results[1], 'label_3': results[2], 'label_4': results[3]}
    df = pd.DataFrame(dict)
    df.to_csv('../output/val1.csv', index=False)
