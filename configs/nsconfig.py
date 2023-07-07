import torch.nn as nn
dataconfig = {
    'audio_path': 'D:/Datasets/NextSpeaker/train_val/',
    'video_path': 'D:/Datasets/NextSpeaker/face_deepface15/train_val/',
    'meta_path': 'D:/Datasets/NextSpeaker/next_speaker_val.csv',
    'batch_size': 12,
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
    'n_class': 4,
    'modality_specific_depth': 11,
    'ftmode': 'multimodal',
    'pretrain_path': '../weights/cav-mae-scale++.pth',
}

vocalistconfig = {
    'pool': True
}

trainconfig = {
    'n_epochs': 50,
    'loss': nn.BCEWithLogitsLoss(),
    'warmup': True,
    'lr': 1e-4,
    'head_lr': 1.0,
    'lr_adapt': False,  # only for cavmae
    'lr_patience': 10,  # only for cavmae
    'lrscheduler_start': 2,
    'lrscheduler_step': 1,
    'lrscheduler_decay': 0.9,
    'exp_dir': 'D:/github/ACMMM2023/logs/',
    'freeze_base': False,   # only for cavmae
    'save_model': True,
    'savename': 'VocaList 15frame nonorm pool4 batch12 1e-4 0.9 threhold0.4',
    'validate_step': 1,
    'cls_threshold': 0.4,
    'audio_norm': False

}
