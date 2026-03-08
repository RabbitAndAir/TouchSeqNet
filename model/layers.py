import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from mamba_ssm import Mamba

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TConv_MultiHeadAttention(nn.Module):
    def __init__(self, d_input, d_model, num_heads):
        super(TConv_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_input, d_model)
        self.k_linear = nn.Linear(d_input, d_model)
        self.v_linear = nn.Linear(d_input, d_model)
        self.out_linear = nn.Linear(d_model, d_input)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        x = x.permute(2, 0, 1)  # [seq_length, batch_size, channels]
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        seq_length, batch_size, embed_dim = Q.size()

        Q = Q.contiguous().view(seq_length, batch_size * self.num_heads, self.d_k).transpose(0, 1)  # [batch_size*num_heads, seq_length, d_k]

        K = K.contiguous().view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)  # [batch_size*num_heads, seq_length, d_k]
        V = V.contiguous().view(-1, batch_size * self.num_heads, self.d_k).transpose(0, 1)  # [batch_size*num_heads, seq_length, d_k]

        attn_output_weights = torch.bmm(Q, K.transpose(1, 2))
        # [batch_size * num_heads,seq_length,kdim] x [batch_size * num_heads, kdim, seq_length]
        # =  [batch_size * num_heads, seq_length, seq_length]  这就num_heads个QK相乘后的注意力矩阵
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, src_len, src_len]

        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, V)
        # Z = [batch_size * num_heads, src_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
        # = # [batch_size * num_heads,src_len,vdim]
        # 这就num_heads个Attention(Q,K,V)结果

        attn_output = attn_output.transpose(0, 1).contiguous().view(batch_size, seq_length, embed_dim)
        # 先transpose成 [src_len, batch_size* num_heads ,kdim]
        # 再view成 [batch_size,src_len,num_heads*kdim]
        attn_output_weights = attn_output_weights.view(batch_size, self.num_heads, seq_length, seq_length)

        Z = self.out_linear(attn_output).permute(0, 2, 1)  # [batch_size, channels, seq_length]
        return Z, attn_output_weights

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if type(x) == list:
            return self.norm(x[1] + self.dropout(self.a * sublayer(x)))
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class MambaBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(MambaBlock, self).__init__()
        # self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.attn = Mamba(
            d_model=d_model,  # Model dimension
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2  # Block expansion factor
        )
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        # Apply attention (replaces self.attn.forward)
        x = self.skipconnect1(x, lambda _x: self.attn(_x))

        # Apply feed-forward network
        x = self.skipconnect2(x, self.ffn)
        return x


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class CrossAttnTRMBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(CrossAttnTRMBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, rep_visible, rep_mask_token, mask=None):
        x = [rep_visible, rep_mask_token]
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x[1], _x[0], _x[0], mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.tconv_multihead_attn = TConv_MultiHeadAttention(n_outputs, n_outputs, 8)
        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        # out = self.net(x)
        # res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out + res)

        # Z, attn_output_weights = self.tconv_multihead_attn(x)
        # out = self.net(Z)
        # res = x if self.downsample is None else self.downsample(x)

        out = self.net(x)
        Z, attn_output_weights = self.tconv_multihead_attn(out)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(Z + res)

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 池化到 [batch_size, channels, 1]
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()  # 检查输入张量的形状 [batch_size, channels, seq_length]

        # 全局池化，得到形状 [batch_size, channels, 1]
        y = self.global_pool(x).view(batch_size, channels)  # 转换为 [batch_size, channels]

        # 通过第一个全连接层，进行降维
        y = self.fc1(y)  # [batch_size, channels // reduction_ratio]
        y = torch.relu(y)

        # 通过第二个全连接层，还原回原始通道数
        y = self.fc2(y)  # [batch_size, channels]
        y = self.sigmoid(y).view(batch_size, channels, 1)  # 恢复形状 [batch_size, channels, 1]

        # 扩展维度并与原始输入相乘，实现通道加权
        return x * y.expand_as(x)

