import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from networks.models import GeneratorTransformer
from datasets.qa_datasets import QADataset
from tokenizers import Tokenizer

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

UNK_INDEX = 0
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

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
             d_feedforward=2048,
             print_model_size=True)
    model = model.to(DEVICE)

    dataset = QADataset("data/chatbot_kor/chatbot.csv", tokenizer, max_length=seq_size, pad_token=PAD_INDEX)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    num_epoch = 10
    for epoch in range(num_epoch):
        progress_bar = tqdm(dataloader, desc="Epoch {}".format(epoch))
        # model.train()
        # for source, target, label in progress_bar:
        #     source = source.to(DEVICE)
        #     target = target.to(DEVICE)
        #     label = label.to(DEVICE)
        #
        #     gen, past_kv = model(source, target)
        #
        #     loss = criterion(gen.transpose(1,2), label)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     progress_bar.set_postfix({
        #         "loss": loss.item()
        #     })

        model.eval()
        past_kv = None
        target = [PAD_INDEX] * seq_size
        target[0] = BOS_INDEX
        target = torch.tensor(target, dtype=torch.long)
        with torch.no_grad():
            source, _, _ = dataset[0]
            source = source.to(DEVICE)
            target = target.to(DEVICE)
            # label = label.to(DEVICE)

            source = source.unsqueeze(0)
            target = target.unsqueeze(0)

            index = 0
            while index < seq_size:
                gen, past_kv = model(source, target, past_kv)
                pred = gen[0, index]
                token_index = torch.argmax(pred)
                target[0, index] = token_index
                if token_index == EOS_INDEX: break

                index += 1

            ids = target[0, :index+1]
            print(tokenizer.decode(ids.numpy()))

        # criterion(gen, target)