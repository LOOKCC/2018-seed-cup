import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwemCat(nn.Module):
    def __init__(self, input_size, embedding_dim, word2vec=None):
        super(SwemCat, self).__init__()
        self.embedding_dim = embedding_dim
        self.word2vec = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        if word2vec is not None:
            self.word2vec.weight.data.copy_(word2vec)


    def forward(self, title, desc, t_len, d_len):
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
                input = input[:seq_len]
                output, _ = input.max(0, keepdim=True)
            else:
                output = torch.zeros((1, self.embedding_dim), device=device)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs

    def swem_avg(self, inputs, seq_lens):
        outputs = []
        for input, seq_len in zip(inputs, seq_lens):
            if seq_len > 0:
                input = input[:seq_len]
                output = input.mean(0, keepdim=True)
            else:
                output = torch.zeros((1, self.embedding_dim), device=device)
            outputs.append(output)
        outputs = torch.cat(outputs)
        return outputs


class SwemHier(nn.Module):
    def __init__(self, input_size, embedding_dim, word2vec=None):
        super(SwemHier, self).__init__()
        self.embedding_dim = embedding_dim
        self.word2vec = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        if word2vec is not None:
            self.word2vec.weight.data.copy_(word2vec)

    def forward(self, title, desc, t_len, d_len):
        title_vec = self.word2vec(title)  # (N, L, D)
        desc_vec = self.word2vec(desc)  # (N, L, D)
        output = torch.cat([self.swem_hier(title_vec, 3),
                            self.swem_hier(desc_vec, 3),
                            self.swem_hier(title_vec, 5),
                            self.swem_hier(desc_vec, 5)], 1)
        return output

    def swem_hier(self, inputs, k):
        outputs = F.avg_pool2d(inputs, (k, 1), stride=1)
        outputs, _ = outputs.max(1)
        return outputs

