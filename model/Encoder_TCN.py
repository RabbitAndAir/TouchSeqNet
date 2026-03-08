import torch
from torch import nn
from model.layers import TemporalBlock, ChannelAttention


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.channel_attn = ChannelAttention(num_channels[-1])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # return self.network(x)
        return self.channel_attn(self.network(x))

class Encoder_TCN(nn.Module):
    def __init__(self, input_projection, encoder, tcn):
        super(Encoder_TCN, self).__init__()
        self.input_projection = input_projection
        self.encoder = encoder  # 使用预训练的 encoder
        self.tcn = tcn  # 使用 TCN 模块进行进一步的处理
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, data1, data2, seq1_mask, seq2_mask):

        x1 = self.input_projection(data1).transpose(1, 2).contiguous()
        x2 = self.input_projection(data2).transpose(1, 2).contiguous()
        # 通过 encoder 提取特征
        en_output1 = self.encoder(x1).transpose(1, 2).contiguous()
        en_output2 = self.encoder(x2).transpose(1, 2).contiguous()

        output1 = self.tcn(en_output1)
        output2 = self.tcn(en_output2)

        output1 = self.global_avg_pool(output1).squeeze(-1)
        output2 = self.global_avg_pool(output2).squeeze(-1)
        return output1, output2


class SiameseClassifier(nn.Module):
    def __init__(self, encoder_tcn, hidden_dim=128):
        super().__init__()
        self.encoder_tcn = encoder_tcn
        self.classifier = nn.Sequential(
            nn.Linear(2 * 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data1, data2, mask1, mask2):
        out1, out2 = self.encoder_tcn(data1, data2, mask1, mask2)
        concat = torch.cat([out1, out2], dim=-1)
        logits = self.classifier(concat)
        return out1, out2, logits

# class SiameseClassifier(nn.Module):
#     def __init__(self, encoder_tcn, hidden_dim=128):
#         super().__init__()
#         self.encoder_tcn = encoder_tcn
#         self.classifier = nn.Sequential(
#             nn.Linear(2 * 128, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#         self.enable_classification = False  # ✅ 添加阶段控制开关
#
#     def forward(self, data1, data2, mask1, mask2):
#         out1, out2 = self.encoder_tcn(data1, data2, mask1, mask2)
#         if self.enable_classification:
#             concat = torch.cat([out1, out2], dim=-1)
#             logits = self.classifier(concat)
#             return out1, out2, logits
#         else:
#             return out1, out2, None  # ✅ 对比学习阶段，不输出 logits
