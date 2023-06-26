import torch.nn as nn
dataconfig = {
    'audio_path': 'D:/Datasets/NextSpeaker/train_val/',
    'video_path': 'D:/Datasets/NextSpeaker/train_val/sample_frames/',
    'meta_path': 'D:/Datasets/NextSpeaker/next_speaker_train.csv',
    'batch_size': 4,
    'num_workers': 2,
    'shuffle': True
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
    'n_class': 1,
    'modality_specific_depth': 11,
    'ftmode': 'multimodal',
    'pretrain_path': '../weights/cav-mae-scale++.pth',
}

trainconfig = {
    'n_epochs': 30,
    'loss': nn.BCEWithLogitsLoss(),
    'warmup': True,
    'lr': 1e-3,
    'head_lr': 10.0,
    'lr_adapt': False,
    'lr_patience': 10,
    'lrscheduler_start': 2,
    'lrscheduler_step': 1,
    'lrscheduler_decay': 0.9,
    'exp_dir': 'D:/github/ACMMM2023/logs/',
    'freeze_base': True,
    'save_model': True,
    'savename': 'cavmae-ns'

}
