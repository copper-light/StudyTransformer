import torch
import torch.nn as nn

from networks import Transformer, make_pad_mask, make_subsequent_mask


PAD = 1

class Generator(nn.Module):
    def __init__(self, n_src_voca, n_tgt_voca, n_seq=256, n_block=6, d_embedding=512, n_heads=8, d_attention=512, d_feedforward=2048):
        super().__init__()

        self.transformer = Transformer(n_src_voca, n_tgt_voca, n_block=n_block, d_embedding=d_embedding, n_heads=n_heads, d_attention=d_attention, d_feedforward=d_feedforward)
        self.fc = nn.Sequential(
            nn.Linear(d_embedding, n_tgt_voca),
            nn.Softmax(dim=-1),
        )

    def forward(self, src, tgt, past_key_values=None):
        # src,tgt: (batch, seq_size)
        src_mask = make_pad_mask(src, src, pad_index=PAD)
        tgt_mask = make_pad_mask(tgt, tgt, pad_index=PAD) & make_subsequent_mask(tgt, tgt)
        cross_mask = make_pad_mask(src, tgt, pad_index=PAD)

        # embedding: (batch, seq_size, embedding_size)

        out = self.transformer(src, tgt, past_key_values=past_key_values, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
        out= self.fc(out)
        return out

