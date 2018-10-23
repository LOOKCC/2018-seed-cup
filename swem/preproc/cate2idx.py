import pickle

train_path = '../../data/train_a.txt'
valid_path = '../../data/valid_a.txt'
save_path = './cate2idx.pkl'

label2idx = [{}, {}, {}]
idx0, idx1, idx2 = 0, 0, 0

with open(train_path, 'r') as fp:
    fp.readline()
    for line in fp.readlines():
        line = line.strip().split('\t')
        if line[5] not in label2idx[0]:
            label2idx[0][line[5]] = idx0
            idx0 += 1
        if line[6] not in label2idx[1]:
            label2idx[1][line[6]] = idx1
            idx1 += 1
        if line[7] not in label2idx[2]:
            label2idx[2][line[7]] = idx2
            idx2 += 1

with open(valid_path, 'r') as fp:
    fp.readline()
    for line in fp.readlines():
        line = line.strip().split('\t')
        if line[5] not in label2idx[0]:
            label2idx[0][line[5]] = idx0
            idx0 += 1
        if line[6] not in label2idx[1]:
            label2idx[1][line[6]] = idx1
            idx1 += 1
        if line[7] not in label2idx[2]:
            label2idx[2][line[7]] = idx2
            idx2 += 1

print('==> Saving cate indices table in {}'.format(save_path))
with open(save_path, 'wb') as fp:
    pickle.dump(label2idx, fp)