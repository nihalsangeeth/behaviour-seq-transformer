import numpy as np
import torch
from torch import nn


class FF(nn.Module):
    """
    Feed-forward in a transformer layer.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.lin_2(self.relu(self.lin_1(x)))
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention block in a transformer layer.
    """

    def __init__(self, att_dim, n_heads):
        super().__init__()
        # Check for compatible  #Attention Heads
        self.n_heads = n_heads
        # Check compatibility for input size and #attention heads.
        assert att_dim % self.n_heads == 0
        self.att_size = int(att_dim / n_heads)

        # Query, Key, Value
        self._query = nn.Linear(att_dim, att_dim, bias=False)
        self._key = nn.Linear(att_dim, att_dim, bias=False)
        self._value = nn.Linear(att_dim, att_dim, bias=False)

        # Attention Block
        self.dense = nn.Linear(att_dim, att_dim, bias=False)
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        scale_factor = torch.sqrt(torch.FloatTensor([self.n_heads])).item()
        batch_size = q.size(0)

        # To Multiple Attention Heads
        _query = self._query(q).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _key = self._key(k).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _value = self._value(v).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)

        # Scaled dot-product Attention score
        score = torch.matmul(_query, _key.transpose(-2, -1)) / scale_factor
        # Mask applied.
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        # Softmax on Score
        score = self.activation(score)
        z = torch.matmul(self.dropout(score), _value)

        # To fully-connected layer
        z = z.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.n_heads)
        return self.dense(z)


class EncoderCell(nn.Module):
    """
    Encoder Cell contains MultiHeadAttention > Add & LayerNorm1 >
    Feed Forward > Add & LayerNorm2
    """

    def __init__(self, input_size, hidden_size, n_heads):
        super().__init__()
        # Attention Block
        self.mh_attention = MultiHeadAttention(input_size, n_heads)
        self.lnorm_1 = nn.LayerNorm(input_size)
        # Feed forward block
        self.ff = FF(input_size, hidden_size)
        self.lnorm_2 = nn.LayerNorm(input_size)
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attention_out = self.mh_attention(x, x, x, mask)
        attention_out = self.lnorm_1(self.dropout(attention_out) + x)

        ff_attention = self.ff(attention_out)
        return self.lnorm_2(self.dropout(ff_attention) + attention_out)


class Encoder(nn.Module):
    """
    Encoder Block with n stacked encoder cells.
    """

    def __init__(self, input_size, hidden_size, n_layers, n_heads):
        super().__init__()
        # Stack of encoder-cells n_layers high
        self.stack = nn.ModuleList()
        # Building encoder stack
        for layer in range(n_layers):
            self.stack.append(EncoderCell(input_size, hidden_size, n_heads))
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        for cell in self.stack:
            x = cell(self.dropout(x), mask)
        return x

