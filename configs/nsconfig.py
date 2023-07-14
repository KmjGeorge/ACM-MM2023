import torch
import torch.nn as nn
import numpy as np

dataconfig = {
    'audio_path': 'D:/Datasets/NextSpeaker/test/',
    'video_path': 'D:/Datasets/NextSpeaker/face_deepface15/test/',
    'meta_path': 'D:/Datasets/NextSpeaker/next_speaker_test.csv',
    'batch_size': 12,
    'num_workers': 2,
    'shuffle': True,
    'audio_mean': None,
    'video_mean': np.array([78.9679, 44.1420, 29.7095]),
    'audio_std': None,
    'video_std': np.array([74.0668, 47.5246, 35.4166]),
}

audioconfig = {
    'skip_norm': False,
    'norm_mean': 0,
    'norm_std': 1,
    'target_length': 1024,
    'melbins': 128,
    'freqm': 0,
    'timem': 0,
    'mixup': False,
    'noise': False,
}

cavmaeconfig = {
    'n_class': 4,
    'modality_specific_depth': 11,
    'ftmode': 'multimodal',
    'pretrain_path': '../weights/cav-mae-scale++.pth',
}

vocalistconfig = {
    'pool': False
}

trainconfig = {
    'n_epochs': 30,
    'loss': nn.BCEWithLogitsLoss(),
    'warmup': True,
    'lr': 1e-4,
    'head_lr': 1.0,
    'lr_adapt': False,  # only for cavmae
    'lr_patience': 15,  # only for cavmae
    'lrscheduler_start': 5,
    'lrscheduler_step': 1,
    'lrscheduler_decay': 0.9,
    'exp_dir': 'D:/github/ACMMM2023/logs/',
    'freeze_base': False,  # only for cavmae
    'save_model': True,
    'savename': 'VocaList 15frame pretrain picnorm=255 melnorm=zscale backward1 concat4 batch12 1e-4 5 0.9 threhold0.5 macro',
    'validate_step': 1,
    'cls_threshold': 0.5,

}
