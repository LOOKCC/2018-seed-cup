#!/usr/bin/env python
# coding=utf-8
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




def get_train_words(train_data):
    train_set = {}
    for line in train_data:
        for word in line[2]:
            train_set[word] = 0
    return train_set


def process(data, args, train_words, label_dict, cate):
    title = []
    label = []
    for line in data:
        temp = [x for x in line[2] if x in train_words]
        title.append(' '.join(temp))
        label.append(label_dict[line[cate]])
    return title, label

def process_test(data, args, train_words):
    title = []
    for line in data:
        temp = [x for x in line[2] if x in train_words]
        title.append(' '.join(temp))
    return title


def get_label_data(test_data, keys):
    if len(keys) == 0:
        return test_data
    elif len(keys) == 1:
        return [x for x in test_data if x[5] == keys[0]]
    elif len(keys) == 2:
        return [x for x in test_data if (x[5] == keys[0] and x[6] == keys[1])]        


def train_test(train_data, test_data, class_info, keys, param, num_round):
    train_words = get_train_words(train_data)
    cate_idx = 0
    if len(keys) == 0:
        key_list = list(class_info.keys())
        cate_idx = 5
    elif len(keys) == 1:
        key_list = list(class_info[keys[0]].keys())
        cate_idx = 6
    elif len(keys) == 2:
        key_list = list(class_info[keys[0]][keys[1]].keys())
        cate_idx = 7
    else:
        print('label error')
        exit(0)
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i

    train_title, train_label = process(train_data, args, train_words, label2idx, cate_idx)
    test_title = process_test(test_data, args, train_words)
    param['num_class'] = len(label2idx)
    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()
    tfidf = vectorizer.fit_transform(train_title)
    # tfidf = tfidftransformer.fit_transform()
    dtrain = xgb.DMatrix(tfidf, label=train_label)
    evallist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    print("cate1 train OK")

    tfidf = vectorizer.transform(test_title)
    # tfidf = tfidftransformer.transform()
    dtest = xgb.DMatrix(tfidf)
    pred = bst.predict(dtest)

    for i in range(len(test_data)):
        test_data[i].append(key_list[int(pred[i])])
    return test_data

    
def main(args):
    finall_test = []
    f = open(args.class_info,'rb')
    class_info = pickle.load(f)
    train_leveled_data, _ = load_leveled_data(args.train_file)
    train_data_1 = get_class_data(train_leveled_data, [])

    # test_leveled_data, _ = load_leveled_data(args.test_file)
    # test_data_1 = get_class_data(test_leveled_data, [])
    test_data_1 = load_data(args.test_file)
    param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
    num_round = 200 # 循环次数
    test_data_1 = train_test(train_data_1, test_data_1,class_info, [], param, num_round)

    for key_1 in class_info.keys():
        train_data_2 = get_class_data(train_leveled_data, [key_1])
        test_data_2 = get_label_data(test_data_1, [key_1])     
        param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
        num_round = 10 # 循环次数
        test_data_2 = train_test(train_data_2, test_data_2, class_info, [key_1], param, num_round)

        for key_2 in class_info[key_1].keys():
            train_data_3 = get_class_data(train_leveled_data, [key_1, key_2])
            test_data_3 = get_label_data(test_data_2, [key_1, key_2])
            param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
            num_round = 5 # 循环次数
            test_data_3 = train_test(train_data_3, test_data_3, class_info, [key_1, key_2], param, num_round)
            finall_test += test_data_3
    
    return finall_test 


def test(test_result):
    predict = [x[5:8] for x in test_result]
    label = [x[8:11] for x in test_result]
    predict = np.array(predict)
    label = np.array(label)

    score = f1_score(predict[:, 0], label[:, 0], average='weighted')
    print('cate1_acc: ', np.mean(predict[:, 0]==label[:, 0]))
    print('cate1_F1 score: ', score)

    score = f1_score(predict[:, 1], label[:, 1], average='weighted')
    print('cate2_acc: ', np.mean(predict[:, 1]==label[:, 1]))
    print('cate2_F1 score: ', score)

    score = f1_score(predict[:, 2], label[:, 2], average='weighted')
    print('cate3_acc: ', np.mean(predict[:, 2]==label[:, 2]))
    print('cate3_F1 score: ', score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Training Data file')
    parser.add_argument('test_file', type=str, help='Testing data file')
    parser.add_argument('class_info', type=str, help='word to idx file, which has deleted the uncommonly uesd words')
    args = parser.parse_args()
    
    test_result = main(args)
    test(test_result)
    # with open("submit.txt", "w") as f:
    #     f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
    #     for x in test_result:
    #         f.write(x[0]+'\t'+str(x[5])+'\t'+str(x[6])+'\t'+str(x[7])+'\n')

    # finall = []
    # with open("submit.txt", 'r') as f:
    #     f.readline()
    #     lines_sub = f.readlines()
    # with open(args.test_file, 'r') as f:
    #     f.readline()
    #     lines_test = f.readlines()
    # for test_line in lines_test:
    #     for sub_line in lines_sub:
    #         if test_line[0:33] == sub_line[0:33]:
    #             finall.append(sub_line)
    # with open("submit_2.txt", 'w') as f:
    #     f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
    #     for line in finall:
    #         f.write(line) 
    

