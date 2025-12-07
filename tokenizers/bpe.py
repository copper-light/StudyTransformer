import unicodedata
from collections import Counter

def to_unicode(text: str):
    text = unicodedata.normalize("NFC", text)
    return text


class BpeTokenizer:

    def __init__(self):
        self.vocabulary = {}

    def train(self, corpus:str):
        # corpus = corpus.strip()
        words = self.preprocess(corpus)

        self.tokens = self.split_charecter(words)

        num_epochs = 10
        epoch = 0
        while len(self.tokens) > 10 or epoch < num_epochs:
            epoch += 1
            pair_freqs = self.count_pairs(self.tokens)
            topk = Counter(pair_freqs).most_common(1)
            if topk[0][1] == 1: break
            self.merge(topk[0][0][0], topk[0][0][1], self.tokens)

        self.tokens = list(set(sum(self.tokens, [])))
        self.vocabulary = set(self.tokens)
        return self.vocabulary

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
        for w in split_characters:
            for i in range(len(w)-1):
                pair = (w[i], w[i+1])
                if pair not in output:
                    output[pair] = 1
                else:
                    output[pair] += 1
        return output

    def merge(self, a, b, target):
        for w in target:
            if len(w) >= 2:
                for i in range(len(w)-2, -1, -1):
                    if w[i] == a and w[i+1] == b:
                        w[i] = a+b
                        w.pop(i+1)


    def tokenize(self, text:str):
        tokens = []
        sentence = self.preprocess(text)
        print(sentence)
        for word in sentence:
            i = 0
            find = 0
            while i < len(word):
                prev = None
                for j in range(i + 1, len(word)+1):
                    search = word[i:j]
                    if search in self.vocabulary:
                        prev = search
                        find = j
                    else:
                        if prev is not None:
                            break

                if prev is not None:
                    tokens.append(prev)
                    i = find
                else:
                    tokens.append('<UNK>')
                    i += 1

        return tokens




if __name__ == "__main__":
    tokenizer = BpeTokenizer()
    c = tokenizer.train("""얼마 안됐는데 썸타는 거 가능?,썸 정도는 언제 타든 상관 없는 거 같아요.
헤어진 남친이랑 다시 썸타기,다시 썸부터 시작해도 좋을 것 같아요.
헤어진 여자친구랑 썸부터 시작할 수 있을까,완전히 똑같은 시작은 힘들겠지요.
헬스장에서 인사하는 사람 맘에 드는데 번호 달라고 할까?,말씀해보세요.
헬스장에서 자주 보는 여자가 보고싶어,한 번 말을 걸어보세요.
헷갈리는데 유지하는 게 좋을까,좋아하면 유지하게 되겠지요.
현실적 문제로 연애 포기해야할듯,잠시 쉬어도 괜찮아요.
혈액형이 성격이랑 무슨 상관이지,상관 없는 거 같아요.
호감을 표현하는 방법,취미와 성향 등이 같음을 어필해보세요.
혼인신고 아직 안 했는데 이혼해도 될까,결심은 빠르면 빠를수록 좋아요.
혼인신고 하러 왔어,법적 부부가 된 걸 축하해요!
혼인신고하니까 마음이 이상해,법적 부부가 된 걸 축하해요!
혼자 노력하는 연애인 거 같아,힘들면 잠시 놓는것도 방법이에요.
혼자 썸타는 기분이야.,직접적이든 간접적이든 의사를 확실히 밝혀보세요.
혼자 좋아하는 것 같아,외로운 상태인 거 같아요.
나만 좋아하는 것 같아,적극적으로 꼬셔보세요.
혼자 좋아하는 이야기 들어 볼래요?,손수건 준비할게요.
혼자 좋아하는 이야기.,힘들겠지만 제게 말해보세요.
혼자 좋아하는게 이렇게 힘든 적은 처음이에요.,사랑은 더 잘하게 되지 않고 다시 영에서 시작하니 모두 처음이겠죠.
혼자가 편하다는 짝남에게 먼저 대쉬해버림.,거절의 뜻은 아니었나요.
혼자가 편하다는 짝녀에게 들이댔음.,혼자가 편하다는 것이 거절의 뜻은 아니었을까요.
혼자만 설레고 혼자서 끝내는 짝사랑 그만할래.,맘고생 많았어요.
화이트데이에 고백할까요?,선물을 주면서 솔직하고 당당하게 고백해보세요.
화장 안했는데 썸남이 영통 걸었어. 어떡해?,화장실 불빛으로 좀 멀리 가리고 해보세요.
확실히 날 좋아하는 걸 아는 남자랑 친구가 될 수 있을까?,그 사람을 위해서는 그러면 안돼요.
안녕하세요. 제 이름은 한동희 입니다. 여자친구랑 혼인 신고 하려고 준비 중이예요""")
    print("vocabulary:", len(tokenizer.vocabulary))
    t = tokenizer.tokenize('안녕하세요. 제 이름은 한동희 입니다. 여자친구랑 혼인 신고 하려고 준비 중이예요')
    print(t)


