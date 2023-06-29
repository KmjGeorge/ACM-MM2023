# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torchvision.transforms as T
from PIL import Image
import PIL
import librosa
import pandas as pd
import h5py
from configs.nsconfig import *
import torchvision
import cv2


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)


def plot_pic(picture, title):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title)
    axs.imshow(picture, aspect="auto")
    plt.show(block=False)


def fbank_h5(save_path):
    assert not os.path.exists(save_path), "文件{}已存在！".format(save_path)
    df = pd.read_csv(dataconfig['meta_path'])
    loop = tqdm(df['id'])
    fbanks = []
    ids = []
    for id in loop:
        audio_full_path = os.path.join(dataconfig['audio_path'], id + '_audio.wav')
        waveform, sr = torchaudio.load(audio_full_path)
        waveform = waveform - waveform.mean()
        # plot_waveform(waveform, sr)
        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        #                                           window_type='hanning', num_mel_bins=audioconfig['melbins'],
        #                                           dither=0.0,
        #                                           frame_shift=10)
        fbank = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                     n_fft=1024,
                                                     win_length=None,
                                                     hop_length=442,
                                                     center=True,
                                                     pad_mode="reflect",
                                                     power=2.0,
                                                     norm="slaney",
                                                     onesided=True,
                                                     n_mels=128,
                                                     mel_scale="htk")(waveform)
        # print(fbank.shape)
        # plot_spectrogram(fbank.squeeze(), sr)
        fbank = torch.transpose(fbank, 1, 2)
        fbank.squeeze_()
        # print(fbank.shape)
        target_length = audioconfig['target_length']
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(audioconfig['freqm'])
        timem = torchaudio.transforms.TimeMasking(audioconfig['timem'])
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if audioconfig['freqm'] != 0:
            fbank = freqm(fbank)
        if audioconfig['timem'] != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if audioconfig['skip_norm'] == False:
            fbank = (fbank - audioconfig['norm_mean']) / (audioconfig['norm_std'])
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass
        if audioconfig['noise'] == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-audioconfig['target_length'], audioconfig['target_length']), 0)
        # print(fbank.shape)
        fbank = np.array(fbank)
        # plot_spectrogram(fbank.T)

        fbanks.append(fbank)
        ids.append(id)
        loop.set_description('读取数据集...')
    print('制作h5数据集中，可能需要数分钟时间...')
    with h5py.File(save_path, 'a') as f:
        f.create_dataset('id', data=ids)
        f.create_dataset('fbanks', data=fbanks)
    print('制作h5数据集完成，保存至 {}'.format(save_path))


def frames_h5(save_path):
    assert not os.path.exists(save_path), "文件{}已存在！".format(save_path)
    df = pd.read_csv(dataconfig['meta_path'])
    loop = tqdm(df['id'])

    for id in loop:
        video_id = [id[:-4] + str(i + 1) for i in range(4)]
        frames = []
        for pos_num in video_id:
            video = []
            for i in range(10):
                frame_full_path = os.path.join(dataconfig['video_path'], 'frame_{}/'.format(i), pos_num + '_video.jpg')
                video_frame = torchvision.io.image.read_image(frame_full_path)  # (3, 224, 224)
                video.append(np.array(video_frame))
            video = np.array(video)  # 每个视频 (10, 3, 224, 224)
            frames.append(video)  # 每个ID对应 (4, 10, 3, 224, 224)
        label1 = df[df['id'] == id]['label_1'].values[0]
        label2 = df[df['id'] == id]['label_2'].values[0]
        label3 = df[df['id'] == id]['label_3'].values[0]
        label4 = df[df['id'] == id]['label_4'].values[0]
        label = [label1, label2, label3, label4]
        with h5py.File(save_path, 'a') as f:
            # print(f['frames'].shape)
            # print(f['labels'].shape)
            # print(np.array(frames).shape)
            # print(np.array(label).shape)
            try:
                f['frames'].resize((f['frames'].shape[0] + 1, f['frames'].shape[1], f['frames'].shape[2],
                                    f['frames'].shape[3], f['frames'].shape[4], f['frames'].shape[5]))
                f['frames'][-1] = frames
                f['labels'].resize((f['labels'].shape[0] + 1, f['labels'].shape[1]))
                f['labels'][-1] = label
                # print(f['frames'].shape, f['labels'].shape)
            except:
                f.create_dataset('frames', chunks=True, maxshape=(None, 4, 10, 3, 224, 224), data=np.array([frames]))
                f.create_dataset('labels', chunks=True, maxshape=(None, 4), data=np.array([label]))
        loop.set_description('读取数据集...')
    print('制作h5数据集完成，保存至 {}'.format(save_path))


class NSDataset(Dataset):
    def __init__(self, audio_h5, frames_h5):
        with h5py.File(audio_h5, 'r') as f:
            self.audio = f['fbanks'][:]
            self.id = f['id'][:]
        with h5py.File(frames_h5, 'r') as f:
            self.label = f['labels'][:]
        self.frames_h5 = frames_h5

    def __len__(self):
        return len(self.audio) * 4

    def __getitem__(self, idx):
        video_idx = int(idx / 4)
        video_idx_offset = int(idx % 4)
        with h5py.File(self.frames_h5, 'r') as f:
            video = f['frames'][video_idx][video_idx_offset]
            video = video.swapaxes(0, 1)  # (C, D, H, W)
        return self.audio[video_idx], video, self.label[video_idx], self.id[video_idx]


class NSDataset_reassemble4(Dataset):
    def __init__(self, audio_h5, frames_h5, method='mean'):
        with h5py.File(audio_h5, 'r') as f:
            self.audio = f['fbanks'][:]
            self.id = f['id'][:]
        with h5py.File(frames_h5, 'r') as f:
            self.label = f['labels'][:]
        self.frames_h5 = frames_h5
        self.method = method

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        with h5py.File(self.frames_h5, 'r') as f:
            video = f['frames'][idx]  # (4, 10, 3, 224, 224)
            video = video.swapaxes(2, 4)  # (4, 10, 224, 224, 3)
            video = video.swapaxes(2, 3)  # (4, 10, 224, 224, 3)
            if self.method == 'mean':
                sum = np.sum(video, axis=0)
                video = (sum / 4).astype(np.uint8)  # (10, 224, 224, 3)
            elif self.method == 'concat':
                downsampled_video = np.ndarray(shape=(4, 10, 112, 112, 3), dtype=np.uint8)
                for i in range(4):
                    for j in range(10):
                        downsample = cv2.resize(video[i][j], dsize=(112, 112), interpolation=cv2.INTER_LINEAR)
                        downsampled_video[i][j] = downsample
                concat1 = np.concatenate((downsampled_video[0], downsampled_video[1]), axis=1)
                concat2 = np.concatenate((downsampled_video[2], downsampled_video[3]), axis=1)
                concat = np.concatenate((concat1, concat2), axis=2)  # (10, 224, 224, 3)
                video = concat
            else:
                raise 'Error method !'
            video = video.swapaxes(1, 3)  # (10, 3, 224, 224)
            video = video.swapaxes(2, 3)  # (10, 3, 224, 224)
            video = video.swapaxes(0, 1)  # (3, 10, 224, 224)
        return self.audio[idx], video, self.label[idx], self.id[idx]


def get_dataloader(reassemble_method=None):
    if reassemble_method:
        if reassemble_method not in ['mean', 'concat']:
            raise 'Error Reassemble Method'
        train = NSDataset_reassemble4('../dataset/train_fbank.h5', '../dataset/train_frames.h5',
                                      method=reassemble_method)
        val = NSDataset_reassemble4('../dataset/val_fbank.h5', '../dataset/val_frames.h5')
    else:
        train = NSDataset('../dataset/train_fbank.h5', '../dataset/train_frames.h5')
        val = NSDataset('../dataset/val_fbank.h5', '../dataset/val_frames.h5')
    train_dataloader = DataLoader(dataset=train, batch_size=dataconfig['batch_size'], shuffle=dataconfig['shuffle'],
                                  num_workers=dataconfig['num_workers'])

    val_dataloader = DataLoader(dataset=val, batch_size=dataconfig['batch_size'], shuffle=False,
                                num_workers=dataconfig['num_workers'])
    return train_dataloader, val_dataloader


def get_testloader(reassemble_method=None):
    if reassemble_method:
        if reassemble_method not in ['mean', 'concat']:
            raise 'Error Reassemble Method'
        test = NSDataset('../dataset/test_fbank.h5', '../dataset/test_frames.h5')
    else:
        test = NSDataset('../dataset/val_fbank.h5', '../dataset/val_frames.h5')
    test_dataloader = DataLoader(dataset=test, batch_size=dataconfig['batch_size'], shuffle=False,
                                 num_workers=dataconfig['num_workers'])
    return test_dataloader


class NSDataset_i3d(Dataset):
    def __init__(self, audio_h5, i3d_path, meta_path, method='mean'):
        with h5py.File(audio_h5, 'r') as f:
            self.audio = f['fbanks'][:]
            self.id = f['id'][:]
        self.i3d_path = i3d_path
        self.frames_h5 = frames_h5
        self.method = method
        self.meta_path = meta_path

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        speaker_id = [self.id[idx].decode()[:-4] + '1', self.id[idx].decode()[:-4] + '2',
                      self.id[idx].decode()[:-4] + '3', self.id[idx].decode()[:-4] + '4']
        video_flow = np.zeros(shape=(4, 4, 1024))
        video_fps = np.zeros(shape=(4,))
        video_rgb = np.zeros(shape=(4, 4, 1024))
        video_timestamps_ms = np.zeros(shape=(4, 4))
        for i in range(len(speaker_id)):
            video_flow[i] = np.load(os.path.join(self.i3d_path, speaker_id[i] + '_video_flow.npy'))  # (4, 1024)
            video_fps[i] = np.load(os.path.join(self.i3d_path, speaker_id[i] + '_video_fps.npy'))  # (,)
            video_rgb[i] = np.load(os.path.join(self.i3d_path, speaker_id[i] + '_video_rgb.npy'))  # (4, 1024)
            video_timestamps_ms[i] = np.load(
                os.path.join(self.i3d_path, speaker_id[i] + '_video_timestamps_ms.npy'))  # (4,

        if self.method == 'mean':
            flow = np.sum(video_flow, axis=0) / 4
            fps = np.sum(video_fps, axis=0) / 4
            rgb = np.sum(video_rgb, axis=0) / 4
            timestamps_ms = np.sum(video_timestamps_ms, axis=0) / 4
        elif self.method == 'concat':
            flow = video_flow.reshape(16, 1024)
            fps = video_fps.reshape(4, )
            rgb = video_rgb.reshape(16, 1024)
            timestamps_ms = video_timestamps_ms.reshape(16, )
        else:
            raise 'Error method !'

        id = self.id[idx].decode()
        df = pd.read_csv(self.meta_path)
        label1 = df[df['id'] == id]['label_1'].values[0]
        label2 = df[df['id'] == id]['label_2'].values[0]
        label3 = df[df['id'] == id]['label_3'].values[0]
        label4 = df[df['id'] == id]['label_4'].values[0]
        label = [label1, label2, label3, label4]
        return self.audio[idx], (flow, fps, rgb, timestamps_ms), label, id


if __name__ == '__main__':
    # h5generator('dataset.h5')
    # fbank_h5('test_fbank.h5')
    # frames_h5('test_frames.h5')
    # from torchvision.datasets import ImageFolder
    #
    # from torch.utils import data
    # from torchvision import transforms
    #
    # my_trans = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # dataset = ImageFolder(frames_folder, transform=my_trans)
    # train_loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    # print(len(train_loader))

    '''
    val = NSDataset_reassemble4('train_fbank.h5', 'train_frames.h5', method='mean')
    ns_dataloader = DataLoader(dataset=val, batch_size=2, shuffle=dataconfig['shuffle'],
                               num_workers=dataconfig['num_workers'])
    loop = tqdm(ns_dataloader)
    index = 1
    for audio, video, y, id in loop:
        print(video.shape)
        if index == 0:
            break
        for i in range(len(audio)):
            plot_spectrogram(audio[i].transpose(0, 1), id[i])
        for i in range(len(video)):
            pic = torch.transpose(video[i][:, 4, :, :], 0, 2)  # 打印第5帧
            pic = torch.transpose(pic, 0, 1)

            plot_pic(pic.numpy(), id[i])
        print(y)
        print(id)
        index -= 1
    '''
    val = NSDataset_i3d('train_fbank.h5', 'D:/Datasets/NextSpeaker/i3d/',
                        meta_path='D:/Datasets/NextSpeaker/next_speaker_train.csv', method='concat')
    ns_dataloader = DataLoader(dataset=val, batch_size=2, shuffle=dataconfig['shuffle'],
                               num_workers=dataconfig['num_workers'])
    index = 0
    for audio, video, y, id in val:
        print('flow', video[0].shape, 'fps', video[1].shape, 'rgb', video[2].shape, 'timestamps', video[3].shape)
        if index > 1:
            break
        index += 1
