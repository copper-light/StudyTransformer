import torch
import torch.nn as nn

from networks.networks import TokenEmbedding, PositionalEncoding, Encoder, Decoder, make_pad_mask, make_subsequent_mask

class GeneratorTransformer(nn.Module):

    def __init__(self, n_src_voca, n_tgt_voca, n_seq=256, n_block=6, d_embedding=512, n_heads=8, d_attention=512, d_feedforward = 2048):
        super().__init__()
        self.src_embedding = TokenEmbedding(vocab_size=n_src_voca, d_embedding=d_embedding)
        self.tgt_embedding = TokenEmbedding(vocab_size=n_tgt_voca, d_embedding=d_embedding)
        self.position_encoding = PositionalEncoding(n_seq=n_seq, d_embedding=d_embedding)
        self.encoder = Encoder(n_block, d_embedding, n_heads, d_attention, d_feedforward)
        self.decoder = Decoder(n_block, d_embedding, n_heads, d_attention, d_feedforward)
        self.generator = nn.Linear(d_embedding, n_tgt_voca)

    def forward(self, src, tgt, past_key_values=None):
        if past_key_values is None:
            src_mask = make_pad_mask(src, src)
            embedded_src = self.src_embedding(src)
            embedded_src = self.position_encoding(embedded_src)
            past_key_values = self.encoder(embedded_src, mask=src_mask)

        cross_mask = make_pad_mask(src, tgt)
        tgt_mask = make_pad_mask(tgt, tgt) & make_subsequent_mask(tgt, tgt)

        embedded_tgt = self.tgt_embedding(tgt)
        embedded_tgt = self.position_encoding(embedded_tgt)
        out = self.decoder(embedded_tgt, past_key_values, self_mask=tgt_mask, cross_mask=cross_mask)
        out = self.generator(out)
        return out, past_key_values



if __name__ == "__main__":
    model = GeneratorTransformer(n_src_voca=100,
                                 n_tgt_voca=100,
                                 n_seq=256,
                                 n_block=6,
                                 d_embedding=512,
                                 n_heads=8,
                                 d_attention=512,
                                 d_feedforward=2048)

    corpus = torch.randint(0, 100, (2, 256))

    x, o = model(corpus,corpus)