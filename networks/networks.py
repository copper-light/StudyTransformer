import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask:torch.Tensor=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk / math.sqrt(query.size(-1)) # d_k: scaling by fc(embedding).output_size
        if mask is not None:
            qk = qk.masked_fill(mask == 0, -1e9)
        prob = nn.functional.softmax(qk, dim=-1)
        return torch.matmul(prob, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_embedding, d_attention_input):
        super().__init__()
        self.n_heads = n_heads
        self.d_attention_input = d_attention_input

        self.input_fc = nn.Linear(d_embedding, 3 * n_heads * d_attention_input)
        self.attention = Attention()
        self.output_fc = nn.Linear(n_heads * d_attention_input, d_attention_input)


    def forward(self, x, mask=None):
        batch_size = x.size(0)
        x = self.input_fc(x) # batch, seq, vector
        x = x.view(batch_size, -1, 3, self.n_heads, self.d_attention_input) # batch, seq, qkv, head, vector
        x = x.permute(0, 2, 3, 1, 4)  # batch, qkv, head, seq, vector
        query = x[:, 0, :, :, :]
        key   = x[:, 1, :, :, :]
        value = x[:, 2, :, :, :]
        x = self.attention(query, key, value, mask=mask) # batch, head, seq, vector
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, self.n_heads * self.d_attention_input)
        x = self.output_fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    seq = torch.randn(3, 10, 5)
    attention = MultiHeadAttention(10, 5, 10)
    print(attention(seq).shape)
