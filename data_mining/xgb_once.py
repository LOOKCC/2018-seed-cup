import xgboost as xgb
import numpy as np
from load_data import load_data
from load_data import get_class_data
from load_data import load_leveled_data
import argparse
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.metrics import f1_score
from scipy.sparse import hstack
import os


def get_train_words(train_data):
    train_set = {}
    for line in train_data:
        for word in line[2]:
            train_set[word] = 0
        for word in line[4]:
            train_set[word] = 0
    return train_set


def process(data, args, train_words, label_dict, cate):
    title = []
    label = []
    for line in data:
        temp = [x for x in line[2] if x in train_words]
        temp += [x for x in line[4] if x in train_words]
        title.append(' '.join(temp))
        label.append(label_dict[line[cate]])
    return title, label


def process_test(data, args, train_words):
    title = []
    for line in data:
        temp = [x for x in line[2] if x in train_words]
        temp += [x for x in line[4] if x in train_words]
        title.append(' '.join(temp))
    return title


def get_train_weight(label):
    class_weight = {}
    label_weight = []
    for i in range(len(label)):
        if label[i] in class_weight:
            class_weight[label[i]] += 1
        else:
            class_weight[label[i]] = 1
    for key in class_weight.keys():
        class_weight[key] = len(label)/class_weight[key] / \
            len(list(class_weight.keys()))

    for i in range(len(label)):
        label_weight.append(class_weight[label[i]])

    return label_weight


def train(args, num_round):
    f = open(args.class_info, 'rb')
    class_info = pickle.load(f)
    train_leveled_data, _ = load_leveled_data(args.train_file)
    train_data_1 = get_class_data(train_leveled_data, [])

    test_leveled_data, _ = load_leveled_data(args.test_file)
    test_data_1 = get_class_data(test_leveled_data, [])

    train_words = get_train_words(train_data_1)
    key_list = []
    for key_1 in class_info.keys():
        for key_2 in class_info[key_1].keys():
            key_list += list(class_info[key_1][key_2])
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i
    print(len(key_list))
    train_title, train_label = process(
        train_data_1, args, train_words, label2idx, 7)
    train_weight = get_train_weight(train_label)
    test_title, test_label = process(
        test_data_1, args, train_words, label2idx, 7)
    vectorizer_1 = CountVectorizer()
    title_train = vectorizer_1.fit_transform(train_title)
    title_test = vectorizer_1.transform(test_title)

    # test_data_1 = load_data(args.test_file)
    param = {'max_depth': 7, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1,
             'objective': 'multi:softmax', 'num_class': len(label2idx)}  # 参数
    dtrain = xgb.DMatrix(title_train, label=train_label, weight=train_weight)
    dtest = xgb.DMatrix(title_test, label=test_label)
    evallist = [(dtest, 'test')]
    file_path = os.path.join(args.save_path, 'cate.pkl')
    if os.path.exists(file_path):
        local_data = pickle.load(open(file_path, 'rb'))
        bst = local_data[0]
        bst = xgb.train(param, dtrain, num_round, evallist,
                        xgb_model=bst, early_stopping_rounds=20)
    else:
        bst = xgb.train(param, dtrain, num_round,
                        evallist, early_stopping_rounds=20)
    # save xgb train_words vectorizer
    to_save = [bst, vectorizer_1, train_words]
    with open(file_path, 'wb') as f:
        pickle.dump(to_save, f)


def valid(args):
    f = open(args.class_info, 'rb')
    class_info = pickle.load(f)

    local_data = pickle.load(
        open(os.path.join(args.save_path, 'cate.pkl'), 'rb'))
    test_data_1 = load_data(args.test_file)

    key_list = []
    for key_1 in class_info.keys():
        for key_2 in class_info[key_1].keys():
            key_list += list(class_info[key_1][key_2])
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i

    test_title = process_test(test_data_1, args, local_data[2])
    title = local_data[1].transform(test_title)
    dtest = xgb.DMatrix(title)
    pred = local_data[0].predict(dtest)

    for i in range(len(test_data_1)):
        test_data_1[i].append(key_list[int(pred[i])])

    return test_data_1


def val(val_result):
    predict = [x[8] for x in val_result]
    label = [x[7] for x in val_result]
    predict = np.array(predict)
    label = np.array(label)

    score_1 = f1_score(predict, label, average='macro')
    print('cate1_acc: ', np.mean(predict == label))
    print('cate1_F1 score: ', score_1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default='../data/train_b.txt', help='Training Data file')
    parser.add_argument('--test_file', type=str,
                        default='../data/valid_b.txt', help='Testing data file')
    parser.add_argument('--save_path', type=str,
                        default='./models', help='Path to save the models')
    parser.add_argument('--class_info', type=str, default='../data/class_info.pkl',
                        help='word to idx file, which has deleted the uncommonly uesd words')
    parser.add_argument('--test', action='store_true', help='train or test')
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch to train')

    args = parser.parse_args()

    if not args.test:
        train(args, args.epoch)
        test_file = args.test_file
        args.test_file = args.train_file
        train_result = valid(args)
        args.test_file = test_file
        test_result = valid(args)
        print('-----train f1-----')
        val(train_result)
        print('-----test f1-----')
        val(test_result)
    else:
        test_result = valid(args)
        print('OK, to save')
        with open("../output/submit.txt", "w") as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for x in test_result:
                f.write(x[0]+'\t'+str(x[5])+'\t'+str(x[6])+'\t'+str(x[7])+'\n')
                # f.write(x[0]+'\t'+str(x[5])+'\t'+str(x[6])+'\t'+str(x[7])+'\t'+str(x[8])+'\t'+str(x[9])+'\t'+str(x[10])+'\n')
        # make the order the same with test_file
        finall = []
        with open("../output/submit.txt", 'r') as f:
            f.readline()
            lines_sub = f.readlines()
        with open(args.test_file, 'r') as f:
            f.readline()
            lines_test = f.readlines()
        for test_line in lines_test:
            for sub_line in lines_sub:
                if test_line[0:33] == sub_line[0:33]:
                    finall.append(sub_line)
        with open("../output/submit_ordered.txt", 'w') as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for line in finall:
                f.write(line)
