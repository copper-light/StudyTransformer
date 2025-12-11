import unicodedata
import os
from tqdm import tqdm
from collections import Counter

from enum import Enum


class TOKEN(Enum):
    UNK = 0
    PAD = 1
    BOS = 2
    EOS = 3


class Vocabulary:

    def __init__(self, special_tokens=('<UNK>', '<PAD>', '<BOS>', '<EOS>')):
        self.special_tokens = special_tokens


    # def save

def to_unicode(text: str):
    text = unicodedata.normalize("NFC", text)
    return text


class BpeTokenizer:

    def __init__(self, special_tokens=('<UNK>', '<PAD>', '<BOS>', '<EOS>'), save_path='save/', load_path=None):
        self.vocabulary = {}
        self.special_tokens = special_tokens
        self.voca_to_index = {}
        self.index_to_voca = {}
        self.filename = 'tokenizer.bpe'
        self.save_path = save_path

        if load_path is not None:
            with open(os.path.join(load_path, self.filename), mode='r', encoding='utf-8') as f:
                for line in f.readlines():
                    if line.find(' ') == -1: break
                    index, token, newline = line[:-1].split(' ')
                    token = token.replace('<\\n>', '\n')
                    self.index_to_voca[index] = token
                    self.voca_to_index[token] = index
                    # self.vocabulary

    def train(self, corpus:list, num_epochs:int=100, vocab_size:int=100):
        # corpus = corpus.strip()
        words = []
        for row in corpus:
            words.append(self.preprocess(row))
        words = sum(words, [])
        tokens = self.split_charecter(words)
        vocabulary = set(sum(tokens, []))

        # progress = tqdm(range(num_epochs))

        for epoch in range(num_epochs):
            if len(vocabulary) <= vocab_size: break
            pair_freqs = self.count_pairs(tokens)
            topk = Counter(pair_freqs).most_common(1)
            if topk[0][1] == 1: break
            self.merge(topk[0][0][0], topk[0][0][1], tokens)
            vocabulary = set(sum(tokens, []))
            print('Epoch {}/{}, Size of Voca {}/{}'.format(epoch + 1, num_epochs, len(vocabulary), vocab_size))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, self.filename), mode='w', encoding='utf-8') as f:
            index = 0
            for t in self.special_tokens:
                f.write(f"{index} {t} \n")
                self.index_to_voca[index] = t
                self.voca_to_index[t] = index
                index += 1

            for t in vocabulary:
                t = t.replace('\n', '<\\n>')
                f.write(f"{index} {t} \n")
                self.index_to_voca[index] = t
                self.voca_to_index[t] = index
                index += 1

        return vocabulary

    def preprocess(self, text):
        corpus = to_unicode(text)
        corpus = corpus.replace("\n", "\n ")
        corpus = corpus.replace("\t", "\t ")
        words = corpus.split(' ')

        for i, w in enumerate(words):
            words[i] = chr(2581) + w
        return words

    def split_charecter(self, words:list):
        split_characters = []
        for w in words:
            s_w = [c for c in w]
            split_characters.append(s_w)
        return split_characters

    def count_pairs(self, split_characters:list):
        output = {}
        progress = tqdm(enumerate(split_characters))
        total = len(split_characters)
        for step, w in progress:
            for i in range(len(w)-1):
                pair = (w[i], w[i+1])
                if pair not in output:
                    output[pair] = 1
                else:
                    output[pair] += 1
            progress.set_description(f'Counting paairs.. {step}/{total}')
        return output

    def merge(self, a, b, target):
        progress = tqdm(enumerate(target))
        total = len(target)
        for step, w in progress:
            if len(w) >= 2:
                for i in range(len(w)-2, -1, -1):
                    if w[i] == a and w[i+1] == b:
                        w[i] = a+b
                        w.pop(i+1)

            progress.set_description(f'Merging pairs.. {step}/{total}')


    def tokenize(self, text:str):
        tokens = []
        sentence = self.preprocess(text)

        for word in sentence:
            i = 0
            while i < len(word):
                find = None
                for j in range(len(word), 0, -1):
                    search = word[i:j]
                    if search in self.voca_to_index:
                        find = search
                        i = j
                        break

                if find is not None:
                    tokens.append(find)
                else:
                    tokens.append(self.special_tokens[TOKEN.UNK.value])
                    i += 1

        return [self.special_tokens[TOKEN.BOS.value]] + tokens + [self.special_tokens[TOKEN.EOS.value]]

    def encode(self, text:str, sequence_size = 256):
        tokens = self.tokenize(text)
        indexes = []
        for token in tokens:
            indexes.append(self.voca_to_index[token])

        if len(indexes) < sequence_size:
            indexes += [TOKEN.PAD.value for i in range(sequence_size-len(indexes))]
        else:
            indexes[-1] = TOKEN.EOS.value

        return indexes

    def decode(self, indexes:list):
        tokens = []
        for index in indexes:
            tokens.append(self.index_to_voca[index])

        return tokens

if __name__ == "__main__":
    tokenizer = BpeTokenizer()
    c = tokenizer.train(["제 이름은 한동희 입니다","여자친구랑 혼인 신고 하려고 준비 중이예요"])
#     tokenizer = BpeTokenizer(load_path="save/")
    print("vocabulary:", len(tokenizer.voca_to_index))
    t = tokenizer.tokenize("""안녕하세요. 제 이름은 한동희 입니다. 여자친구랑 혼인 신고 하려고 준비 중이예요""")
    print(t)

    t = tokenizer.encode("""안녕하세요. 제 이름은 한동희 입니다. 여자친구랑 혼인 신고 하려고 준비 중이예요""")
    print(t)

    t = tokenizer.decode(t)
    print(t)


