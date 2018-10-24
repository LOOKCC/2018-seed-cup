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



def process(data, label_dict, cate):
    title = [data[i][1] for i in range(len(data))]
    label = [label_dict[data[i][cate]] for i in range(len(data))]
    return title, label

def process_test(data):
    title = [data[i][1] for i in range(len(data))]
    return title


def get_label_data(test_data, keys, args):
    if args.test:
        cate1 = 2
        cate2 = 3
    else:
        cate1 = 5
        cate2 = 6
    if len(keys) == 0:
        return test_data
    elif len(keys) == 1:
        return [x for x in test_data if x[cate1] == keys[0]]
    elif len(keys) == 2:
        return [x for x in test_data if (x[cate1] == keys[0] and x[cate2] == keys[1])]        


def train_test(train_data, test_data, class_info, keys, param, num_round):
    cate_idx = 0
    if len(keys) == 0:
        key_list = list(class_info.keys())
        cate_idx = 2
    elif len(keys) == 1:
        key_list = list(class_info[keys[0]].keys())
        cate_idx = 3
    elif len(keys) == 2:
        key_list = list(class_info[keys[0]][keys[1]].keys())
        cate_idx = 4
    else:
        print('label error')
        exit(0)
    label2idx = {}
    for i in range(len(key_list)):
        label2idx[key_list[i]] = i

    train_title, train_label = process(train_data, label2idx, cate_idx)
    test_title = process_test(test_data)
    param['num_class'] = len(label2idx)

    dtrain = xgb.DMatrix(train_title, label=train_label)
    evallist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    print("cate1 train OK")

    dtest = xgb.DMatrix(test_title)
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

    test_data_1 = load_data(args.test_file)
    param = {'max_depth':8, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
    num_round = 300 # 循环次数
    test_data_1 = train_test(train_data_1, test_data_1,class_info, [], param, num_round)

    for key_1 in class_info.keys():
        train_data_2 = get_class_data(train_leveled_data, [key_1])
        test_data_2 = get_label_data(test_data_1, [key_1], args)     
        param = {'max_depth':7, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
        num_round = 300 # 循环次数
        test_data_2 = train_test(train_data_2, test_data_2, class_info, [key_1], param, num_round)

        for key_2 in class_info[key_1].keys():
            train_data_3 = get_class_data(train_leveled_data, [key_1, key_2])
            test_data_3 = get_label_data(test_data_2, [key_1, key_2], args)
            param = {'max_depth':7, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':0}  # 参数
            num_round = 400 # 循环次数
            test_data_3 = train_test(train_data_3, test_data_3, class_info, [key_1, key_2], param, num_round)
            finall_test += test_data_3
    
    return finall_test 


def val(val_result):
    predict = [x[2:5] for x in val_result]
    label = [x[5:8] for x in val_result]
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
    
    result = main(args)
    if not args.test:
        val(result)
    else:
        with open("../output/submit.txt", "w") as f:
            f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
            for x in result:
                f.write(x[0]+'\t'+str(x[2])+'\t'+str(x[3])+'\t'+str(x[4])+'\n')
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
        

