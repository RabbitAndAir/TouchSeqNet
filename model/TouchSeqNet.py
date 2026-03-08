import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_

from model.layers import CrossAttnTRMBlock, PositionalEmbedding, TransformerBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

        # TRMs with Mamba-based attention
        # self.TRMs = nn.ModuleList(
        #     [MambaBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)]
        # )

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1,
                                   dim))  # 将 (batch_size, seq_len, d_model) 展开为 (batch_size * seq_len, d_model)，然后映射到 vocab_size --> (batch_size * seq_len, vocab_size)
        ret = F.gumbel_softmax(probs)  # 使用 Gumbel-Softmax 计算离散的样本
        indexes = ret.max(-1, keepdim=True)[1]  # 获取每个时间步的最大值索引（即 token 的索引）-->  ( batch_size * seq_len, 1)
        return indexes.view(bs, length)  # 将索引重新调整为 (batch_size, seq_len) 的形状


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class TouchSeqNet(nn.Module):
    def __init__(self, args):
        super(TouchSeqNet, self).__init__()
        d_model = args.d_model

        self.momentum = args.momentum
        self.linear_proba = True
        self.device = args.device
        self.data_shape = args.data_shape
        self.wave_length = args.wave_length
        self.max_len = int(self.data_shape[1] / args.wave_length)
        print(self.max_len)
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, d_model)

        self.mask_token = nn.Parameter(torch.randn(d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[0], d_model, kernel_size=args.wave_length,
                                          stride=args.wave_length)
        self.encoder = Encoder(args)
        self.momentum_encoder = Encoder(args)
        self.tokenizer = Tokenizer(d_model, args.vocab_size)
        self.reg = Regressor(d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        # self.reg = cross_mamba(d_model=d_model, d_state=16)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x, data_mask):  # x --> (batch_size, channel, seq_len)
        x = self.input_projection(x).transpose(1, 2).contiguous()  # (batch_size, new_seq_len, d_model)
        tokens = self.tokenizer(x)  # (batch_size, new_seq_len)
        x += self.position(x)
        rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + self.position(x)

        index = np.arange(x.shape[1])  # 创建了一个 从 0 到 seq_len - 1 的数组
        random.shuffle(index)  # 随机打乱 index 数组的顺序
        v_index = index[:-self.mask_len]  # 选择可见索引
        m_index = index[-self.mask_len:]  # 不可见索引

        visible = x[:, v_index, :]
        mask = x[:, m_index, :]
        tokens = tokens[:, m_index]
        rep_mask_token = rep_mask_token[:, m_index, :]
        # 掩码选取
        visible_mask = data_mask[:, v_index]
        mask_mask = data_mask[:, m_index]

        rep_visible = self.encoder(visible)  # (batch_size, visible_len, d_model)
        with torch.no_grad():
            # rep_mask = self.encoder(mask)
            rep_mask = self.momentum_encoder(mask)
        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens
