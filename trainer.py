import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from networks.models import GeneratorTransformer
from datasets.qa_datasets import QADataset
from tokenizers import Tokenizer

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

if __name__ == '__main__':
    tokenizer = Tokenizer.from_file("voca.json")
    voca_size = tokenizer.get_vocab_size()
    print("Vocab size: {}".format(voca_size))
    seq_size = 256

    model = GeneratorTransformer(n_src_voca=voca_size,
             n_tgt_voca=voca_size,
             n_seq=seq_size,
             n_block=6,
             d_embedding=512,
             n_heads=8,
             d_attention=512,
             d_feedforward=2048)
    model = model.to(DEVICE)

    dataset = QADataset("data/chatbot_kor/chatbot.csv", tokenizer, max_length=seq_size, pad_token=0)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    past_kv = None

    num_epoch = 10
    for epoch in range(num_epoch):
        progress_bar = tqdm(dataloader, desc="Epoch {}".format(epoch))
        for source, target, label in progress_bar:
            source = source.to(DEVICE)
            target = target.to(DEVICE)
            label = label.to(DEVICE)
            
            gen, past_kv = model(source, target)

            loss = criterion(gen.transpose(1,2), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({
                "loss": loss.item()
            })

        # criterion(gen, target)