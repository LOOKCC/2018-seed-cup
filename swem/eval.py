import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import pickle

import argparse

from sklearn.metrics import f1_score

from .models.cate1_classifier import Cate1Classifier
from .models.cate2_classifier import Cate2Classifier
from .models.cate3_classifier import Cate3Classifier
from .utils.dataset import TrainDataset, padding

WORDS_CNT = 34835
CHARS_CNT = 3939

CATE1_CNT = 10
CATE2_CNT = 64
CATE3_CNT = 125

TRAIN_SAMPLES = 140562

embedding_dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid_feature_path = './preproc/valid_words.pkl'
valid_cate_path = './preproc/valid_cate.pkl'

def eval(clf1, clf2, clf3, dataloader):
    clf1.eval()
    clf2.eval()
    clf3.eval()
    pred1, pred2, pred3 = [], [], []
    target1, target2, target3 = [], [], []
    for title, desc, cate1, cate2, cate3, t_len, d_len in dataloader:
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
        target1.extend(cate1.tolist())
        target2.extend(cate2.tolist())
        target3.extend(cate3.tolist())
    cate1_score = f1_score(target1, pred1, average='macro')
    cate2_score = f1_score(target2, pred2, average='macro')
    cate3_score = f1_score(target3, pred3, average='macro')
    score = 0.1 * cate1_score + 0.3 * cate2_score + 0.6 * cate3_score
    print('Validation cate1 f1 score: {:.4}'.format(cate1_score))
    print('Validation cate2 f1 score: {:.4}'.format(cate2_score))
    print('Validation cate3 f1 score: {:.4}'.format(cate3_score))
    print('Validation weighted f1 score: {:.4}'.format(score))

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf1', required=True, help='cate1 classifier')
    parser.add_argument('--clf2', required=True, help='cate2 classifier')
    parser.add_argument('--clf3', required=True, help='cate3 classifier')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size, default=64')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_cmd()

    with open(valid_feature_path, 'rb') as fp:
        features = pickle.load(fp)
    title = [feature[0] for feature in features]
    desc = [feature[1] for feature in features]
    t_len = [len(feature[0]) for feature in features]
    d_len = [len(feature[1]) for feature in features]


    with open(valid_cate_path, 'rb') as fp:
        cate = pickle.load(fp)
    cate1 = [ca[0] for ca in cate]
    cate2 = [ca[1] for ca in cate]
    cate3 = [ca[2] for ca in cate]

    title = padding(title, max(t_len))
    desc = padding(desc, max(d_len))
    t_len = torch.tensor(t_len)
    d_len = torch.tensor(d_len)
    cate1 = torch.tensor(cate1)
    cate2 = torch.tensor(cate2)
    cate3 = torch.tensor(cate3)

    valid_set = TrainDataset(title, desc, cate1, cate2, cate3, t_len, d_len)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)

    clf1_state = torch.load(args.clf1)
    clf1 = Cate1Classifier(clf1_state['args'])
    clf1.load_state_dict(clf1_state['model'])

    with open('./preproc/mask.pkl', 'rb') as fp:
        mask1, mask2 = pickle.load(fp)

    clf2_state = torch.load(args.clf2)
    clf2 = Cate2Classifier(clf2_state['args'], mask1=mask1)
    clf2.load_state_dict(clf2_state['model'])

    clf3_state = torch.load(args.clf3)
    clf3 = Cate3Classifier(clf3_state['args'], mask2=mask2)
    clf3.load_state_dict(clf3_state['model'])

    clf1 = clf1.to(device)
    clf2 = clf2.to(device)
    clf3 = clf3.to(device)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    with torch.no_grad():
        eval(clf1, clf2, clf3, valid_loader)

