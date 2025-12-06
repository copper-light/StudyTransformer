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
    qk = torch.matmul(query, key.transpose(-2, -1)) # batch, head, q_seq, k_seq
    qk = qk / math.sqrt(query.size(-1)) # d_k: scaling by fc(embedding).output_size
    if mask is not None:
        qk = qk.masked_fill(mask == 0, -1e9)
    prob = nn.functional.softmax(qk, dim=-1)
    return torch.matmul(prob, value) # batch, head, q_seq, k_seq


# attention 값 계산할 때, softmax 취하기 전에 pad 값들이 확률을 취하는 것을 방지하기 위해서 mask 값을 계산함
# 이 때, query, key 값은 임베딩되기 전의 토큰 인덱스 값임 (batch, seq_size)
def make_pad_mask(query, key, pad_index=1):
    query_seq_size = query.size(1)
    key_seq_size = key.size(1)
    q_mask = query.ne(pad_index).unsqueeze(1).unsqueeze(-1) # batch, 1(head), q_seq, 1
    q_mask = q_mask.repeat(1, 1, 1, key_seq_size)
    k_mask = key.ne(pad_index).unsqueeze(1).unsqueeze(1) # batch, 1(head), 1, k_seq
    k_mask = k_mask.repeat(1, 1, query_seq_size, 1)
    mask = q_mask * k_mask
    mask.requires_grad = False
    return mask

def make_subsequent_mask(query, key):
    query_seq_size = query.size(1)
    key_seq_size = key.size(1)

    mask = torch.ones(query.size(0), query_seq_size, key_seq_size).to(query.device)
    mask = mask.tril(diagonal=0).type(torch.bool)
    mask.requires_grad = False
    return mask


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
    def __init__(self, n_block, d_embedding, n_heads, d_attention, d_feedforward):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(EncoderBlock(d_embedding, n_heads, d_attention, d_feedforward))


    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_embedding, n_heads,  d_attention, d_feedforward):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_embedding, n_heads, d_attention)
        self.cross_attention = MultiHeadAttention(d_embedding, n_heads, d_attention, is_cross_attention=True)

        # Position-wise Feed-Forward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_embedding, d_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(d_feedforward, d_embedding),
        )
        self.residual = ResidualBlock(d_embedding)

    def forward(self, x, kv, self_mask=None, cross_mask=None):
        out = self.self_attention(x, mask=self_mask)
        x = self.residual(x, out)

        out = self.cross_attention(x, kv, mask=cross_mask)
        x = self.residual(x, out)

        out = self.feedforward(x)
        x = self.residual(x, out)
        return x


class Decoder(nn.Module):
    def __init__(self, n_block, d_embedding, n_heads, d_attention, d_feedforward):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(DecoderBlock(d_embedding, n_heads, d_attention, d_feedforward))

    def forward(self, x, kv, self_mask=None, cross_mask=None):
        for block in self.blocks:
            x = block(x, kv, self_mask=self_mask, cross_mask=cross_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_seq, d_embedding):
        super().__init__()
        self.n_seq = n_seq
        self.d_embedding = d_embedding
        encoding = torch.zeros(n_seq, d_embedding)
        position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0) # batch, seq, emd
        self.register_buffer('encoding', encoding)

    def forward(self, x):
         return x + self.encoding[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_embedding):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embedding)
        self.d_embedding = d_embedding

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_embedding)


class Transformer(nn.Module):
    def __init__(self, n_src_voca, n_tgt_voca, n_seq=256, n_block=6, d_embedding=512, n_heads=8, d_attention=512, d_feedforward = 2048):
        super().__init__()
        self.src_embedding = TokenEmbedding(vocab_size=n_src_voca, d_embedding=d_embedding)
        self.tgt_embedding = TokenEmbedding(vocab_size=n_tgt_voca, d_embedding=d_embedding)
        self.position_encoding = PositionalEncoding(n_seq=n_seq, d_embedding=d_embedding)
        self.encoder = Encoder(n_block, d_embedding, n_heads, d_attention, d_feedforward)
        self.decoder = Decoder(n_block, d_embedding, n_heads, d_attention, d_feedforward)
        self.fc = nn.Sequential(
            nn.Linear(d_embedding, n_tgt_voca),
            nn.Softmax(dim=-1),
        )

    def forward(self, src, tgt, past_key_values=None):
        if past_key_values is None:
            src_mask = make_pad_mask(src, src)
            src = self.src_embedding(src)
            src = self.position_encoding(src)

            past_key_values = self.encoder(src, mask=src_mask)

        tgt_mask = make_pad_mask(tgt, tgt)
        cross_mask = make_subsequent_mask(src, tgt)

        tgt = self.tgt_embedding(tgt)
        tgt = self.position_encoding(tgt)
        out = self.decoder(tgt, past_key_values, self_mask=tgt_mask, cross_mask=cross_mask)
        out = self.fc(out)
        return out, past_key_values


if __name__ == '__main__':
    vocab_size = 10


    t = Transformer(vocab_size, vocab_size)

    src = (torch.rand(1, 10) * 1000 % vocab_size).type(torch.int)
    tgt = (torch.rand(1, 10) * 1000 % vocab_size).type(torch.int)

    out, past_key_values = t(src, tgt)
    out, past_key_values = t(src, tgt, past_key_values=past_key_values)
    out, past_key_values = t(src, tgt, past_key_values=past_key_values)
    out, past_key_values = t(src, tgt, past_key_values=past_key_values)

    print(out.shape, out)