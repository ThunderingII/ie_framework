import math

import torch

from torch import nn


def gelu(x):
    return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, batch_x):
        x = self.linear1(batch_x)
        x = gelu(x)
        if self.dropout:
            return self.dropout_layer(self.linear2(x))
        else:
            return self.linear2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout=0.0):
        super().__init__()
        assert hidden_size % head_num == 0, 'h must be divided evenly'
        self.dk = hidden_size // head_num
        self.head_num = head_num
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, batch_x, mask_x):
        Q = self.query(batch_x)
        K = self.key(batch_x)
        V = self.value(batch_x)

        # hs = head_num*dk == hidden_size
        bs, seq_len, hs = Q.size()
        # change to [bs, head_num, seq_len, dk]
        Q = Q.view(bs, seq_len, self.head_num, self.dk).permute(0, 2, 1, 3)
        K = K.view(bs, seq_len, self.head_num, self.dk).permute(0, 2, 1, 3)
        V = V.view(bs, seq_len, self.head_num, self.dk).permute(0, 2, 1, 3)
        # qk [bs, head_num, seq_len, seq_len]
        qk = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dk) + mask_x
        proba = nn.Softmax(dim=-1)(qk)

        if self.dropout:
            proba = self.dropout_layer(proba)

        value = torch.matmul(proba, V)
        # change to [bs, seq_len, head_num, dk]
        value = value.permute(0, 2, 1, 3)
        # view as [bs, seq_len, hs]
        value = value.contiguous().view(bs, seq_len, hs)
        if self.dropout:
            return self.dropout_layer(self.dense(value))
        else:
            return self.dense(value)


class TransformerEncode(nn.Module):
    def __init__(self, hidden_size, head_num, dropout=0.0, direction=False,
                 left2right=True):
        super().__init__()
        self.direction = direction
        self.left2right = left2right
        self.mha = MultiHeadAttention(hidden_size, head_num, dropout)
        self.mha_bn = LayerNorm(hidden_size)

        self.fw = FeedForward(hidden_size, dropout)
        self.fw_bn = LayerNorm(hidden_size)

    def forward(self, batch_x, mask_x):
        """
        :param batch_x:
        :param mask_x: [bs,1,1,seq_len]
        :return:
        """
        # batch_x must be shape of [bs,seq_len,embed_size]
        # change to [bs,head_num,from_seq,to_seq],[bs,1,1,seq_len]
        mask_x = mask_x.unsqueeze(1).unsqueeze(2)
        if self.direction:
            len_w = mask_x.size()[-1]
            one = torch.ones((len_w, len_w), dtype=torch.int32)
            if self.left2right:
                mask_atten = (1 - torch.triu(one, diagonal=1)).unsqueeze(
                    0).unsqueeze(1)
            else:
                mask_atten = torch.triu(one).unsqueeze(0).unsqueeze(1)
            mask_x = mask_x.int() & mask_atten
        mask_x = ((1 - mask_x) * -10000.0).float()
        x = self.mha(batch_x, mask_x) + batch_x
        x = self.mha_bn(x)
        x = self.fw(x) + x
        return self.fw_bn(x), mask_x
