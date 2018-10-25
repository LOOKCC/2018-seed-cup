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
from scipy.sparse import hstack



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


    
def train(args):
    f = open(args.class_info,'rb')
    class_info = pickle.load(f)
    train_leveled_data, _ = load_leveled_data(args.train_file)
    train_data_1 = get_class_data(train_leveled_data, [])

    train_words = get_train_words(train_data_1)    
    key_list = list(class_info.keys())
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i
    train_title, train_label = process(train_data_1, args, train_words, label2idx, 5)
    vectorizer_1 = CountVectorizer()
    title = vectorizer_1.fit_transform(train_title)

    # test_data_1 = load_data(args.test_file)
    param = {'max_depth':8, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':len(label2idx)}  # 参数
    num_round = 300 # 循环次数
    dtrain = xgb.DMatrix(title, label=train_label)
    evallist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    # save xgb train_words vectorizer
    to_save = [bst, vectorizer_1, train_words]
    with open('cate1.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    

    for key_1 in class_info.keys():
        train_data_2 = get_class_data(train_leveled_data, [key_1])
        train_words = get_train_words(train_data_2)    
        key_list = list(class_info[key_1].keys())
        label2idx = {}
        for i in range(len(key_list)):
            label2idx[key_list[i]] = i
        train_title, train_label = process(train_data_2, args, train_words, label2idx, 6)
        vectorizer_2 = CountVectorizer()
        title = vectorizer_2.fit_transform(train_title)

        # test_data_2 = get_label_data(test_data_1, [key_1], args)     
        param = {'max_depth':7, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':len(label2idx)}  # 参数
        num_round = 300 # 循环次数
        dtrain = xgb.DMatrix(title, label=train_label)
        evallist  = [(dtrain,'train')]
        bst = xgb.train(param, dtrain, num_round, evallist)
        to_save = [bst, vectorizer_2, train_words]
        with open('cate2_'+str(key_1)+'.pkl', 'wb') as f:
            pickle.dump(to_save, f)

        for key_2 in class_info[key_1].keys():
            train_data_3 = get_class_data(train_leveled_data, [key_1, key_2])
            train_words = get_train_words(train_data_3)    
            key_list = list(class_info[key_1][key_2].keys())
            label2idx = {}
            for i in range(len(key_list)):
                label2idx[key_list[i]] = i
            train_title, train_label = process(train_data_3, args, train_words, label2idx, 7)
            vectorizer_3 = CountVectorizer()
            title = vectorizer_3.fit_transform(train_title)

            # test_data_2 = get_label_data(test_data_1, [key_1], args)     
            param = {'max_depth':7, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':len(label2idx)}  # 参数
            num_round = 300 # 循环次数
            dtrain = xgb.DMatrix(title, label=train_label)
            evallist  = [(dtrain,'train')]
            bst = xgb.train(param, dtrain, num_round, evallist)
            to_save = [bst, vectorizer_3, train_words]
            with open('cate3_'+str(key_1)+'_'+str(key_2)+'.pkl', 'wb') as f:
                pickle.dump(to_save, f)



def valid(args):
    finall = []
    f = open(args.class_info,'rb')
    class_info = pickle.load(f)

    local_data = pickle.load(open('cate1.pkl', 'rb'))
    test_data_1 = load_data(args.test_file)
    key_list = list(class_info.keys())
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i
    test_title = process_test(test_data_1, args, local_data[2])
    title = local_data[1].transform(test_title)
    dtest = xgb.DMatrix(title)
    pred = local_data[0].pred(dtest)

    for i in range(len(test_data_1)):
        test_data_1[i].append(key_list[int(pred[i])])


    for key_1 in class_info.keys():
        local_data = pickle.load(open('cate2_'+str(key_1)+'.pkl', 'rb'))
        test_data_2 = get_label_data(test_data_1, [key_1], args)
        key_list = list(class_info[key_1].keys())
        label2idx = {}
        for i in range(len(key_list)):
            label2idx[key_list[i]] = i
        test_title = process_test(test_data_2, args, local_data[2])
        title = local_data[1].transform(test_title)
        dtest = xgb.DMatrix(title)
        pred = local_data[0].pred(dtest)

        for i in range(len(test_data_2)):
            test_data_2[i].append(key_list[int(pred[i])])


        for key_2 in class_info[key_1].keys():
            local_data = pickle.load(open('cate3_'+str(key_1)+'_'+str(key_2)+'.pkl', 'rb'))
            test_data_3 = get_label_data(test_data_2, [key_1, key_2], args)
            key_list = list(class_info[key_1][key_2].keys())
            label2idx = {}
            for i in range(len(key_list)):
                label2idx[key_list[i]] = i
            test_title = process_test(test_data_3, args, local_data[2])
            title = local_data[1].transform(test_title)
            dtest = xgb.DMatrix(title)
            pred = local_data[0].pred(dtest)

            for i in range(len(test_data_3)):
                test_data_3[i].append(key_list[int(pred[i])])

            finall += test_data_3
    return finall










def val(val_result):
    predict = [x[5:8] for x in val_result]
    label = [x[8:11] for x in val_result]
    predict = np.array(predict)
    label = np.array(label)

    score = f1_score(predict[:, 0], label[:, 0], average='macro')
    print('cate1_acc: ', np.mean(predict[:, 0]==label[:, 0]))
    print('cate1_F1 score: ', score)

    score = f1_score(predict[:, 1], label[:, 1], average='macro')
    print('cate2_acc: ', np.mean(predict[:, 1]==label[:, 1]))
    print('cate2_F1 score: ', score)

    score = f1_score(predict[:, 2], label[:, 2], average='macro')
    print('cate3_acc: ', np.mean(predict[:, 2]==label[:, 2]))
    print('cate3_F1 score: ', score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='Training Data file')
    parser.add_argument('--test_file', type=str, help='Testing data file')
    parser.add_argument('--class_info', type=str, help='word to idx file, which has deleted the uncommonly uesd words')
    parser.add_argument('--test', action='store_true', help='train or test')

    args = parser.parse_args()
    
    train(args)
    result = valid(args)
    if not args.test:
        val(result)
    else:
        with open("../output/submit.txt", "w") as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for x in result:
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
        

