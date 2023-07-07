import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from model.module.vocalist.transformer_encoder import TransformerEncoder
from model.module.vocalist.conv import Conv2d, Conv3d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SyncTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(SyncTransformer, self).__init__()
        self.d_model = d_model
        layers = [32, 64, 128, 256, 512]
        self.vid_prenet = nn.Sequential(
            Conv3d(3, layers[0], kernel_size=7, stride=1, padding=3),

            Conv3d(layers[0], layers[1], kernel_size=5, stride=(1, 2, 1), padding=(1, 1, 2)),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[1], layers[2], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[2], layers[3], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[3], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),

            Conv3d(layers[4], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=(0, 0, 1)),
            Conv3d(layers[4], layers[4], kernel_size=1, stride=1, padding=0), )
        self.aud_prenet = nn.Sequential(
            Conv2d(1, layers[0], kernel_size=3, stride=1, padding=1),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[0], layers[0], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[0], layers[1], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[1], layers[2], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[2], layers[3], kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(layers[3], layers[4], kernel_size=3, stride=1, padding=(0, 1)),
            Conv2d(layers[4], layers[4], kernel_size=1, stride=1, padding=0), )

        self.av_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        self.va_transformer = TransformerEncoder(embed_dim=d_model,
                                                 num_heads=8,
                                                 layers=4,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.25,
                                                 attn_mask=True)
        self.mem_transformer = TransformerEncoder(embed_dim=d_model,
                                                  num_heads=8,
                                                  layers=4,
                                                  attn_dropout=0.0,
                                                  relu_dropout=0.1,
                                                  res_dropout=0.1,
                                                  embed_dropout=0.25,
                                                  attn_mask=True)

        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 4)

    def forward(self, frame_seq, mel_seq, pool=False):
        B = frame_seq.shape[0]
        # print(frame_seq.shape)  # (B, 4, 15, 96, 96)

        aud_embedding = self.aud_prenet(mel_seq)
        aud_embedding = aud_embedding.squeeze(2)
        aud_embedding = aud_embedding.permute(2, 0, 1).contiguous()

        v_speakers = []
        index = np.random.choice([0, 1, 2, 3], size=4, replace=False)  # 打乱四个视频
        v_backward = False  # 仅第一次前向传播需要更新梯度
        for i in index:
            speaker_frame = frame_seq[:, i, :, :, :]
            # print('speaker_frame.shape', speaker_frame.shape)
            # print(speaker_frame.view(B, -1, 3, 48, 96).permute(0, 2, 3, 4, 1).contiguous().type())

            if v_backward:
                with torch.no_grad():
                    vid_embedding = self.vid_prenet(
                        speaker_frame.view(B, -1, 3, 48, 96).permute(0, 2, 3, 4, 1).contiguous())
                    vid_embedding = vid_embedding.squeeze(2).squeeze(2)
                    vid_embedding = vid_embedding.permute(2, 0, 1).contiguous()
            else:
                vid_embedding = self.vid_prenet(
                    speaker_frame.view(B, -1, 3, 48, 96).permute(0, 2, 3, 4, 1).contiguous())
                vid_embedding = vid_embedding.squeeze(2).squeeze(2)
                vid_embedding = vid_embedding.permute(2, 0, 1).contiguous()
                v_backward = True

            # print('embeddding', vid_embedding.shape)     # (10, B, 512)
            v_speakers.append(vid_embedding)
        if pool:
            vid_embedding_reassemble = (v_speakers[0] + v_speakers[1] + v_speakers[2] + v_speakers[3]) / 4  # 平均4个视频的特征
        else:
            vid_embedding_reassemble = torch.cat(
                (v_speakers[0], v_speakers[1], v_speakers[2], v_speakers[3]))  # 拼接4个视频的特征
        # print('4x', vid_embedding_reassemble.shape)      # (40, B ,512)

        av_embedding = self.av_transformer(aud_embedding, vid_embedding_reassemble, vid_embedding_reassemble)
        vid_embedding_reassemble = self.va_transformer(vid_embedding_reassemble, aud_embedding, aud_embedding)

        tranformer_out = self.mem_transformer(av_embedding, vid_embedding_reassemble, vid_embedding_reassemble)
        t = av_embedding.shape[0]

        out = F.max_pool1d(tranformer_out.permute(1, 2, 0).contiguous(), t).squeeze(-1)
        h_pooled = self.activ1(self.fc(out))  # [batch_size, d_model]
        logits_clsf = (self.classifier(h_pooled))
        return logits_clsf.squeeze(-1)


# Test Model
if __name__ == "__main__":
    from torchsummary import summary

    mel_seq = torch.rand([6, 1, 80, 1103])
    frame_seq = torch.rand([6, 4, 15, 96, 96])
    model = SyncTransformer()
    output = model(frame_seq, mel_seq)
    print(output.shape)
    summary(model, input_size=[(4, 15, 96, 96), (1, 80, 1103)], device='cpu')
    # key: state_dict, optimizer, global_step, global_epoch
    weights = torch.load('../weights/VocaLiST_Weights/vocalist_5f_lrs2.pth')
    # for k in weights.keys():
    #     print(k)
    model.load_state_dict(weights['state_dict'])
    print(count_parameters(model))
