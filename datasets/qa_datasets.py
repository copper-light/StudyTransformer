import csv
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

def add_padding(ids, max_length=256, pad_value=1):
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + ([pad_value] * (max_length - len(ids)))

    return ids


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer:Tokenizer, max_length=256, pad_token=None):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.data_path = data_path
        self.max_length = max_length
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data.append((row[0], row[1]))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data[idx]
        q = add_padding(self.tokenizer.encode(row[0]).ids, self.max_length, self.pad_token)
        a = add_padding(self.tokenizer.encode(row[1]).ids, self.max_length, self.pad_token)
        return torch.tensor(q, dtype=torch.int), torch.tensor(a, dtype=torch.int)


if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('../voca.json')
    dataset = QADataset('../data/chatbot_kor/chatbot.csv', tokenizer, max_length=256, pad_token=0)
    print(len(dataset))
    print(dataset[0])

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for q, a in loader:
        print(q.shape, a.shape)
        break
