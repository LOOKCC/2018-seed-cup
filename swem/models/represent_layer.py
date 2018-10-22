import torch
import torch.nn as nn

WORDS_CNT = 34835
CHARS_CNT = 3939

CATE1_CNT = 10
CATE2_CNT = 64
CATE3_CNT = 125

TRAIN_SAMPLES = 140562

embedding_dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_dim, args, embedding_weight=None):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.args = args
        if embedding_weight is not None:
            self.embedding.weight = nn.Parameter(embedding_weight)

    def forward(self, inputs, length, mode):
        if length > 0:
            inputs = inputs[:length]
            if self.args.drop and mode and length > 10:
                idx = torch.zeros(length.item()).uniform_()
                return self.embedding(inputs[idx > self.args.drop_rate])
            else:
                return self.embedding(inputs)
        else:
            return torch.zeros((1, self.embedding_dim), device=device)


class SwemCat(nn.Module):
    def __init__(self, args, word2vec=None):
        super(SwemCat, self).__init__()
        self.word2vec = Embedding(WORDS_CNT, embedding_dim, args, word2vec)

    def forward(self, title, desc, t_len, d_len, mode):
        title_vec = self.word2vec(title, t_len, mode)
        desc_vec = self.word2vec(desc, d_len, mode)
        output = torch.cat([self.swem_max(title_vec),
                            self.swem_max(desc_vec),
                            self.swem_avg(title_vec),
                            self.swem_avg(desc_vec)], 1)
        return output

    def swem_max(self, input):
        try:
            output, _ = input.max(0, True)
        except:
            print(input)
            raise
        return output

    def swem_avg(self, input):
        output = input.mean(0, True)
        return output
