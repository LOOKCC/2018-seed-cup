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


class SwemCat(nn.Module):
    def __init__(self, word2vec=None):
        super(SwemCat, self).__init__()
        self.word2vec = nn.Embedding(WORDS_CNT+1, embedding_dim, padding_idx=0)
        if word2vec is not None:
            self.word2vec.weight = nn.Parameter(word2vec)


    def forward(self, title, desc, t_len, d_len, mode):
        title_vec = self.word2vec(title)         # (N, L, D)
        desc_vec = self.word2vec(desc)           # (N, L, D)
        output = torch.cat([self.swem_max(title_vec, t_len),
                            self.swem_max(desc_vec, d_len),
                            self.swem_avg(title_vec, t_len),
                            self.swem_avg(desc_vec, d_len)], 1)
        return output

    def swem_max(self, inputs, seq_lens):
        outputs = []
        for input, seq_len in zip(inputs, seq_lens):
            if seq_len > 0:
                input = input[:seq_len].data
                output, _ = input.max(0, keepdim=True)
            else:
                output = torch.zeros((1, embedding_dim), device=device)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs

    def swem_avg(self, inputs, seq_lens):
        outputs = []
        for input, seq_len in zip(inputs, seq_lens):
            if seq_len > 0:
                input = input[:seq_len].data
                output = input.mean(0, keepdim=True)
            else:
                output = torch.zeros((1, embedding_dim), device=device)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs
