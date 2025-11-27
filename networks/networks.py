import math
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_channel, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(n_channel)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y):
        x = self.norm(x + y)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def calc_attention(query, key, value, mask:torch.Tensor=None):
    qk = torch.matmul(query, key.transpose(-2, -1))
    qk = qk / math.sqrt(query.size(-1)) # d_k: scaling by fc(embedding).output_size
    if mask is not None:
        qk = qk.masked_fill(mask == 0, -1e9)
    prob = nn.functional.softmax(qk, dim=-1)
    return torch.matmul(prob, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embedding, n_heads, d_attention, is_cross_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_attention = d_attention
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.input_kv = nn.Linear(d_embedding, 2 * n_heads * d_attention)
            self.input_q = nn.Linear(d_embedding, n_heads * d_attention)
        else:
            self.input_fc = nn.Linear(d_embedding, 3 * n_heads * d_attention)

        self.output_fc = nn.Linear(n_heads * d_attention, d_embedding)

    def forward(self, x, kv=None, mask=None):
        batch_size = x.size(0)

        if self.is_cross_attention and kv is not None:
            q = self.input_q(x)  # batch, seq, vector
            q = q.view(batch_size, -1, self.n_heads, self.d_attention)
            q = q.transpose(1, 2)

            kv = self.input_kv(kv)  # batch, seq, vector
            kv = kv.view(batch_size, -1, 2, self.n_heads, self.d_attention)  # batch, seq, qkv, head, vector
            kv = kv.permute(0, 2, 3, 1, 4)  # batch, kv, head, seq, vector
            k = kv[:, 0, :, :, :]
            v = kv[:, 1, :, :, :]
        else:
            x = self.input_fc(x) # batch, seq, vector
            x = x.view(batch_size, -1, 3, self.n_heads, self.d_attention) # batch, seq, qkv, head, vector
            x = x.permute(0, 2, 3, 1, 4)  # batch, qkv, head, seq, vector
            q = x[:, 0, :, :, :]
            k = x[:, 1, :, :, :]
            v = x[:, 2, :, :, :]

        x = calc_attention(q, k, v, mask=mask) # batch, head, seq, vector
        x = x.transpose(1, 2) # batch, seq, head, vector
        x = x.contiguous().view(batch_size, -1, self.n_heads * self.d_attention)
        x = self.output_fc(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_embedding, n_heads,  d_attention, d_feedforward):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_embedding, n_heads, d_attention)

        # Position-wise Feed-Forward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_embedding, d_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(d_feedforward, d_embedding),
        )
        self.residual = ResidualBlock(d_embedding)

    def forward(self, x, mask=None):
        out = self.self_attention(x, mask=mask)
        x = self.residual(x, out)

        out = self.feedforward(x)
        x = self.residual(x, out)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layer, d_embedding, n_heads, d_attention, d_feedforward):
        self.super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(EncoderBlock(d_embedding, n_heads, d_attention, d_feedforward))


    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_embedding, n_heads,  d_attention, d_feedforward):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_embedding, n_heads, d_attention)

        self.cross_attention = MultiHeadAttention(d_embedding, n_heads, d_attention, is_cross_attention=True)

        # Position-wise Feed-Forward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_embedding, d_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(d_feedforward, d_embedding),
        )
        self.residual = ResidualBlock(d_embedding)

    def forward(self, x, kv, mask=None):
        out = self.masked_attention(x, mask=mask)
        x = self.residual(x, out)

        out = self.cross_attention(x, kv, mask=mask)
        x = self.residual(x, out)

        out = self.feedforward(x)
        x = self.residual(x, out)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layer, d_embedding, n_heads, d_attention, d_feedforward):
        self.super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(DecoderBlock(d_embedding, n_heads, d_attention, d_feedforward))

    def forward(self, x, kv, mask=None):
        for block in self.blocks:
            x = block(x, kv, mask)
        return x

#
# BOS = 1
# EOS = 2
# PAD = 3
# UNK = 4

# class Transformer(nn.Module):
#     def __init__(self, n_layer, n_seq, d_embedding, n_heads, d_attention, d_feedforward):
#         super().__init__()
#         self.n_seq = n_seq
#         self.encoder = Encoder(n_layer, d_embedding, n_heads, d_attention, d_feedforward)
#         self.decoder = Decoder(n_layer, d_embedding, n_heads, d_attention, d_feedforward)
#
#     def forward(self, x, mask=None):
#         encoder_out = self.encoder(x, mask=mask)
#         out = torch.Tensor(PAD).expand_as(x)
#         for i in range(self.n_seq):
#             x = self.decoder(x, encoder_out, mask)
#
#             torch.
#             if x == EOS:
#                 break
#         return out

if __name__ == '__main__':
    seq = torch.randn(3, 10, 5)
    attention = MultiHeadAttention(5, 20, 10)
    print(attention(seq).shape)

    encoder = EncoderBlock(d_embedding=5, n_heads=10, d_attention=20, d_feedforward=40)
    out = encoder(seq)
    print(out.shape)

    decoder = DecoderBlock(d_embedding=5, n_heads=10, d_attention=20, d_feedforward=40)
    print(decoder(seq, out).shape)