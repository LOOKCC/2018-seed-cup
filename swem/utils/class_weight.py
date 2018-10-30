import torch
import pickle

CATE1_CNT = 20
CATE2_CNT = 135
CATE3_CNT = 265

TRAIN_SAMPLES = 911256

with open('../preproc/train_cate.pkl', 'rb') as fp:
    cate = pickle.load(fp)

cate1_weight = torch.zeros(CATE1_CNT)
cate2_weight = torch.zeros(CATE2_CNT)
cate3_weight = torch.zeros(CATE3_CNT)

for i in range(TRAIN_SAMPLES):
    cate1_weight[cate[i][0]] += 1
    cate2_weight[cate[i][1]] += 1
    cate3_weight[cate[i][2]] += 1

cate1_weight = TRAIN_SAMPLES / cate1_weight / CATE1_CNT
cate2_weight = TRAIN_SAMPLES / cate2_weight / CATE2_CNT
cate3_weight = TRAIN_SAMPLES / cate3_weight / CATE3_CNT

with open('class_weight.pkl', 'wb') as fp:
    pickle.dump((cate1_weight, cate2_weight, cate3_weight), fp)