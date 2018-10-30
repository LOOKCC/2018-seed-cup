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


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        def exp_minmax(x): return np.exp(x - np.max(x))

        def denom(x): return 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def process_test_char(data, args, train_words):
    title = []
    for line in data:
        temp = [x for x in line[1] if x in train_words]
        temp += [x for x in line[3] if x in train_words]
        title.append(' '.join(temp))
    return title


def process_test_word(data, args, train_words):
    title = []
    for line in data:
        temp = [x for x in line[2] if x in train_words]
        temp += [x for x in line[4] if x in train_words]
        title.append(' '.join(temp))
    return title


def get_label_data(test_data, keys, args):
    if args.test:
        cate1 = 5
        cate2 = 6
    else:
        cate1 = 8
        cate2 = 9
    if len(keys) == 0:
        return test_data
    elif len(keys) == 1:
        return [x for x in test_data if x[cate1] == keys[0]]
    elif len(keys) == 2:
        return [x for x in test_data if (x[cate1] == keys[0] and x[cate2] == keys[1])]


def valid(args):
    finall = []
    f = open(args.class_info, 'rb')
    class_info = pickle.load(f)

    local_data_word = pickle.load(
        open(os.path.join(args.word_save_path, 'cate1.pkl'), 'rb'))
    local_data_char = pickle.load(
        open(os.path.join(args.char_save_path, 'cate1.pkl'), 'rb'))

    test_data_1 = load_data(args.test_file)
    key_list = list(class_info.keys())
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i
    test_title_char = process_test_char(test_data_1, args, local_data_char[2])
    test_title_word = process_test_word(test_data_1, args, local_data_word[2])

    title_char = local_data_char[1].transform(test_title_char)
    title_word = local_data_word[1].transform(test_title_word)
    dtest_char = xgb.DMatrix(title_char)
    dtest_word = xgb.DMatrix(title_word)
    pred_char = local_data_char[0].predict(dtest_char, output_margin=True)
    pred_word = local_data_word[0].predict(dtest_word, output_margin=True)

    pred_char = softmax(pred_char)
    pred_word = softmax(pred_word)
    x = np.argmax(pred_char + pred_word, axis=1)

    for i in range(len(test_data_1)):
        test_data_1[i].append(key_list[int(x[i])])

    for key_1 in class_info.keys():

        local_data_char = pickle.load(
            open(os.path.join(args.char_save_path, 'cate2_'+str(key_1)+'.pkl'), 'rb'))
        local_data_word = pickle.load(
            open(os.path.join(args.word_save_path, 'cate2_'+str(key_1)+'.pkl'), 'rb'))

        test_data_2 = get_label_data(test_data_1, [key_1], args)
        key_list = list(class_info[key_1].keys())
        label2idx = {}
        for i in range(len(key_list)):
            label2idx[key_list[i]] = i
        test_title_char = process_test_char(
            test_data_2, args, local_data_char[2])
        test_title_word = process_test_word(
            test_data_2, args, local_data_word[2])

        title_char = local_data_char[1].transform(test_title_char)
        title_word = local_data_word[1].transform(test_title_word)
        dtest_char = xgb.DMatrix(title_char)
        dtest_word = xgb.DMatrix(title_word)
        pred_char = local_data_char[0].predict(dtest_char, output_margin=True)
        pred_word = local_data_word[0].predict(dtest_word, output_margin=True)

        pred_char = softmax(pred_char)
        pred_word = softmax(pred_word)
        x = pred_char + pred_word
        x = x.reshape(len(x), -1)
        x = np.argmax(x, axis=1)

        for i in range(len(test_data_2)):
            test_data_2[i].append(key_list[int(x[i])])

        for key_2 in class_info[key_1].keys():
            local_data_word = pickle.load(
                open(os.path.join(args.word_save_path, 'cate3_'+str(key_1)+'_'+str(key_2)+'.pkl'), 'rb'))
            local_data_char = pickle.load(
                open(os.path.join(args.char_save_path, 'cate3_'+str(key_1)+'_'+str(key_2)+'.pkl'), 'rb'))
            test_data_3 = get_label_data(test_data_2, [key_1, key_2], args)
            key_list = list(class_info[key_1][key_2].keys())
            label2idx = {}
            for i in range(len(key_list)):
                label2idx[key_list[i]] = i
            test_title_char = process_test_char(
                test_data_3, args, local_data_char[2])
            test_title_word = process_test_word(
                test_data_3, args, local_data_word[2])

            title_char = local_data_char[1].transform(test_title_char)
            title_word = local_data_word[1].transform(test_title_word)
            dtest_char = xgb.DMatrix(title_char)
            dtest_word = xgb.DMatrix(title_word)
            # x = local_data_char[0].predict(dtest_char)
            pred_char = local_data_char[0].predict(
                dtest_char, output_margin=True)
            pred_word = local_data_word[0].predict(
                dtest_word, output_margin=True)

            pred_char = softmax(pred_char)
            pred_word = softmax(pred_word)
            x = pred_char + pred_word
            x = x.reshape(len(x), -1)
            # print(x.shape)
            x = np.argmax(x, axis=1)

            for i in range(len(test_data_3)):
                test_data_3[i].append(key_list[int(x[i])])

            finall += test_data_3
    return finall


def val(val_result):
    predict = [x[5:8] for x in val_result]
    label = [x[8:11] for x in val_result]
    predict = np.array(predict)
    label = np.array(label)

    score_1 = f1_score(predict[:, 0], label[:, 0], average='macro')
    print('cate1_acc: ', np.mean(predict[:, 0] == label[:, 0]))
    print('cate1_F1 score: ', score_1)

    score_2 = f1_score(predict[:, 1], label[:, 1], average='macro')
    print('cate2_acc: ', np.mean(predict[:, 1] == label[:, 1]))
    print('cate2_F1 score: ', score_2)

    score_3 = f1_score(predict[:, 2], label[:, 2], average='macro')
    print('cate3_acc: ', np.mean(predict[:, 2] == label[:, 2]))
    print('cate3_F1 score: ', score_3)
    print('finall score: '+str(0.1*score_1+0.3*score_2+0.6*score_3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default='../data/train_b.txt', help='Training Data file')
    parser.add_argument('--test_file', type=str,
                        default='../data/valid_b.txt', help='Testing data file')
    parser.add_argument('--word_save_path', type=str,
                        default='./models_weighted', help='Path to save the models')
    parser.add_argument('--char_save_path', type=str,
                        default='./models_weighted_char', help='Path to save the models')
    parser.add_argument('--class_info', type=str, default='../data/class_info.pkl',
                        help='word to idx file, which has deleted the uncommonly uesd words')
    parser.add_argument('--test', action='store_true', help='train or test')
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch to train')

    args = parser.parse_args()

    if not args.test:
        # test_file = args.test_file
        # args.test_file = args.train_file
        # train_result = valid(args)
        # args.test_file = test_file
        test_result = valid(args)
        # print('-----train f1-----')
        # val(train_result)
        print('-----test f1-----')
        val(test_result)
    else:
        test_result = valid(args)
        print('OK, to save')
        with open("../output/bagging.txt", "w") as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for x in test_result:
                f.write(x[0]+'\t'+str(x[5])+'\t'+str(x[6])+'\t'+str(x[7])+'\n')
                # f.write(x[0]+'\t'+str(x[5])+'\t'+str(x[6])+'\t'+str(x[7])+'\t'+str(x[8])+'\t'+str(x[9])+'\t'+str(x[10])+'\n')
        # make the order the same with test_file
        finall = []
        with open("../output/bagging.txt", 'r') as f:
            f.readline()
            lines_sub = f.readlines()
        with open(args.test_file, 'r') as f:
            f.readline()
            lines_test = f.readlines()
        for test_line in lines_test:
            for sub_line in lines_sub:
                if test_line[0:33] == sub_line[0:33]:
                    finall.append(sub_line)
        with open("../output/bagging_ordered.txt", 'w') as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for line in finall:
                f.write(line)
