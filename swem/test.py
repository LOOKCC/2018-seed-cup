import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import pickle

import argparse

from models.cate1_classifier import Cate1Classifier
from models.cate2_classifier import Cate2Classifier
from models.cate3_classifier import Cate3Classifier
from utils.dataset import TestDataset, padding

WORDS_CNT = 72548

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_feature_path = './preproc/test_words.pkl'
test_path = '../data/test_a.txt'


def test(clf1, clf2, clf3, dataloader):
    clf1.eval()
    clf2.eval()
    clf3.eval()
    pred1, pred2, pred3 = [], [], []
    for title, desc, t_len, d_len in dataloader:
        title = title.to(device)
        desc = desc.to(device)
        output1 = clf1(title, desc, t_len, d_len, mode=0)
        output1 = output1.argmax(1)
        pred1.extend(output1.tolist())
        output2 = clf2(title, desc, t_len, d_len, output1, mode=0)
        output2 = output2.argmax(1)
        pred2.extend(output2.tolist())
        output3 = clf3(title, desc, t_len, d_len, output2, mode=0)
        pred3.extend(output3.argmax(1).tolist())
    return pred1, pred2, pred3

def save_result(save_path, preds):
    ids = []
    with open(test_path, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            line = line.strip().split('\t')
            ids.append(line[0])
    with open('./preproc/cate2idx.pkl', 'rb') as fp:
        cate2idx = pickle.load(fp)
    idx2cate = [{cate2idx[j][l]: l for l in cate2idx[j]} for j in range(3)]
    print('==> Saving test result in {}'.format(save_path))
    with open(save_path, 'w') as fp:
        fp.write('id\tcate1\tcate2\tcate3\n')
        for id, p1, p2, p3 in zip(ids, preds[0], preds[1], preds[2]):
            fp.write('{}\t{}\t{}\t{}\n'.format(id, idx2cate[0][p1], idx2cate[1][p2], idx2cate[2][p3]))

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf1', required=True,
                        help='cate1 classifier')
    parser.add_argument('--clf2', required=True,
                        help='cate2 classifier')
    parser.add_argument('--clf3', required=True,
                        help='cate3 classifier')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size, default=64')
    parser.add_argument('--save_path', required=True,
                        help='path to save test result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_cmd()

    with open(test_feature_path, 'rb') as fp:
        features = pickle.load(fp)
    title = [feature[0] for feature in features]
    valid_desc = [feature[1] for feature in features]
    t_len = [len(feature[0]) for feature in features]
    d_len = [len(feature[1]) for feature in features]

    title = padding(title, max(t_len))
    desc = padding(valid_desc, max(d_len))
    t_len = torch.tensor(t_len)
    d_len = torch.tensor(d_len)

    test_set = TestDataset(title, desc, t_len, d_len)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=4)

    clf1_state = torch.load(args.clf1)
    clf1 = Cate1Classifier(WORDS_CNT+1, clf1_state['args'])
    clf1.load_state_dict(clf1_state['model'])

    with open('./preproc/mask.pkl', 'rb') as fp:
        mask1, mask2 = pickle.load(fp)

    clf2_state = torch.load(args.clf2)
    clf2 = Cate2Classifier(WORDS_CNT+1, clf2_state['args'], mask1=mask1)
    clf2.load_state_dict(clf2_state['model'])

    clf3_state = torch.load(args.clf3)
    clf3 = Cate3Classifier(WORDS_CNT+1, clf3_state['args'], mask2=mask2)
    clf3.load_state_dict(clf3_state['model'])

    clf1 = clf1.to(device)
    clf2 = clf2.to(device)
    clf3 = clf3.to(device)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    with torch.no_grad():
        pred1, pred2, pred3 = test(clf1, clf2, clf3, test_loader)

    save_result(args.save_path, (pred1, pred2, pred3))

