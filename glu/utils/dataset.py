import torch
import torch.utils.data as data


class TrainDataset(data.Dataset):
    def __init__(self, title, desc, cate1, cate2, cate3, t_len, d_len):
        self.title = title
        self.desc = desc
        self.cate1 = cate1
        self.cate2 = cate2
        self.cate3 = cate3
        self.t_len = t_len
        self.d_len = d_len

    def __getitem__(self, idx):
        return self.title[idx], self.desc[idx], \
               self.cate1[idx], self.cate2[idx], self.cate3[idx], \
               self.t_len[idx], self.d_len[idx]

    def __len__(self):
        return self.title.size(0)


class EvalDataset(data.Dataset):
    def __init__(self, w_title, w_desc, c_title, c_desc,
                 cate1, cate2, cate3,
                 w_t_len, w_d_len, c_t_len, c_d_len):
        self.w_title = w_title
        self.w_desc = w_desc
        self.c_title = c_title
        self.c_desc = c_desc
        self.cate1 = cate1
        self.cate2 = cate2
        self.cate3 = cate3
        self.w_t_len = w_t_len
        self.w_d_len = w_d_len
        self.c_t_len = c_t_len
        self.c_d_len = c_d_len

    def __getitem__(self, idx):
        return self.w_title[idx], self.w_desc[idx], self.c_title[idx], self.c_desc[idx], \
               self.cate1[idx], self.cate2[idx], self.cate3[idx], \
               self.w_t_len[idx], self.w_d_len[idx], self.c_t_len[idx], self.c_d_len[idx]

    def __len__(self):
        return self.w_title.size(0)


class TestDataset(data.Dataset):
    def __init__(self, w_title, w_desc, c_title, c_desc,
                 w_t_len, w_d_len, c_t_len, c_d_len):
        self.w_title = w_title
        self.w_desc = w_desc
        self.c_title = c_title
        self.c_desc = c_desc
        self.w_t_len = w_t_len
        self.w_d_len = w_d_len
        self.c_t_len = c_t_len
        self.c_d_len = c_d_len

    def __getitem__(self, idx):
        return self.w_title[idx], self.w_desc[idx], self.c_title[idx], self.c_desc[idx], \
               self.w_t_len[idx], self.w_d_len[idx], self.c_t_len[idx], self.c_d_len[idx]

    def __len__(self):
        return self.w_title.size(0)


def padding(seq, max_len):
    pad_seq = torch.zeros((len(seq), max_len), dtype=torch.long)
    for pad_s, s in zip(pad_seq, seq):
        pad_s[:len(s)] = torch.tensor(s)
    return pad_seq
