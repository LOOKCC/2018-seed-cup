import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle

import argparse

from sklearn.metrics import f1_score

from models.cate1_classifier import Cate1Classifier
from models.cate2_classifier import Cate2Classifier
from models.cate3_classifier import Cate3Classifier
from utils.dataset import EvalDataset, padding


WORDS_CNT = 72548
CHARS_CNT = 4933

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid_words_path = './preproc/valid_words.pkl'
valid_chars_path = './preproc/valid_chars.pkl'
valid_cate_path = './preproc/valid_cate.pkl'

def eval(w_clf1, w_clf2, w_clf3, c_clf1, c_clf2, c_clf3, dataloader):
    w_clf1.eval()
    w_clf2.eval()
    w_clf3.eval()
    c_clf1.eval()
    c_clf2.eval()
    c_clf3.eval()
    pred1, pred2, pred3 = [], [], []
    target1, target2, target3 = [], [], []
    for w_title, w_desc, c_title, c_desc, \
        cate1, cate2, cate3, \
        w_t_len, w_d_len, c_t_len, c_d_len in dataloader:
        w_title = w_title.to(device)
        w_desc = w_desc.to(device)
        c_title = c_title.to(device)
        c_desc = c_desc.to(device)
        output1 = F.softmax(w_clf1(w_title, w_desc, w_t_len, w_d_len, mode=0), 1) + \
                  F.softmax(c_clf1(c_title, c_desc, c_t_len, c_d_len, mode=0), 1)
        output1 = output1.argmax(1)
        pred1.extend(output1.tolist())
        output2 = F.softmax(w_clf2(w_title, w_desc, w_t_len, w_d_len, output1, mode=0), 1) + \
                  F.softmax(c_clf2(c_title, c_desc, c_t_len, c_d_len, output1, mode=0), 1)
        output2 = output2.argmax(1)
        pred2.extend(output2.tolist())
        output3 = F.softmax(w_clf3(w_title, w_desc, w_t_len, w_d_len, output2, mode=0), 1) + \
                  F.softmax(c_clf3(c_title, c_desc, c_t_len, c_d_len, output2, mode=0), 1)
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
    parser.add_argument('--w_clf1', required=True,
                        help='word cate1 classifier')
    parser.add_argument('--w_clf2', required=True,
                        help='word cate2 classifier')
    parser.add_argument('--w_clf3', required=True,
                        help='word cate3 classifier')
    parser.add_argument('--c_clf1', required=True,
                        help='char cate1 classifier')
    parser.add_argument('--c_clf2', required=True,
                        help='char cate2 classifier')
    parser.add_argument('--c_clf3', required=True,
                        help='char cate3 classifier')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size, default=64')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_cmd()

    with open(valid_words_path, 'rb') as fp:
        features = pickle.load(fp)
    w_title = [feature[0] for feature in features]
    w_desc = [feature[1] for feature in features]
    w_t_len = [len(feature[0]) for feature in features]
    w_d_len = [len(feature[1]) for feature in features]

    with open(valid_chars_path, 'rb') as fp:
        features = pickle.load(fp)
    c_title = [feature[0] for feature in features]
    c_desc = [feature[1] for feature in features]
    c_t_len = [len(feature[0]) for feature in features]
    c_d_len = [len(feature[1]) for feature in features]


    with open(valid_cate_path, 'rb') as fp:
        cate = pickle.load(fp)
    cate1 = [ca[0] for ca in cate]
    cate2 = [ca[1] for ca in cate]
    cate3 = [ca[2] for ca in cate]

    w_title = padding(w_title, max(w_t_len))
    w_desc = padding(w_desc, max(w_d_len))
    w_t_len = torch.tensor(w_t_len)
    w_d_len = torch.tensor(w_d_len)

    c_title = padding(c_title, max(c_t_len))
    c_desc = padding(c_desc, max(c_d_len))
    c_t_len = torch.tensor(c_t_len)
    c_d_len = torch.tensor(c_d_len)

    cate1 = torch.tensor(cate1)
    cate2 = torch.tensor(cate2)
    cate3 = torch.tensor(cate3)

    valid_set = EvalDataset(w_title, w_desc, c_title, c_desc,
                            cate1, cate2, cate3,
                            w_t_len, w_d_len, c_t_len, c_d_len)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)

    w_clf1_state = torch.load(args.w_clf1)
    w_clf1 = Cate1Classifier(WORDS_CNT+1, w_clf1_state['args'])
    w_clf1.load_state_dict(w_clf1_state['model'])

    c_clf1_state = torch.load(args.c_clf1)
    c_clf1 = Cate1Classifier(CHARS_CNT + 1, c_clf1_state['args'])
    c_clf1.load_state_dict(c_clf1_state['model'])

    with open('./preproc/mask.pkl', 'rb') as fp:
        mask1, mask2 = pickle.load(fp)

    w_clf2_state = torch.load(args.w_clf2)
    w_clf2 = Cate2Classifier(WORDS_CNT+1, w_clf2_state['args'], mask1=mask1)
    w_clf2.load_state_dict(w_clf2_state['model'])

    c_clf2_state = torch.load(args.c_clf2)
    c_clf2 = Cate2Classifier(CHARS_CNT + 1, c_clf2_state['args'], mask1=mask1)
    c_clf2.load_state_dict(c_clf2_state['model'])

    w_clf3_state = torch.load(args.w_clf3)
    w_clf3 = Cate3Classifier(WORDS_CNT+1, w_clf3_state['args'], mask2=mask2)
    w_clf3.load_state_dict(w_clf3_state['model'])

    c_clf3_state = torch.load(args.c_clf3)
    c_clf3 = Cate3Classifier(CHARS_CNT + 1, c_clf3_state['args'], mask2=mask2)
    c_clf3.load_state_dict(c_clf3_state['model'])

    w_clf1 = w_clf1.to(device)
    w_clf2 = w_clf2.to(device)
    w_clf3 = w_clf3.to(device)

    c_clf1 = c_clf1.to(device)
    c_clf2 = c_clf2.to(device)
    c_clf3 = c_clf3.to(device)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    with torch.no_grad():
        eval(w_clf1, w_clf2, w_clf3, c_clf1, c_clf2, c_clf3, valid_loader)

