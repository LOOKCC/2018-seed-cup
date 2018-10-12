#!/usr/bin/env python
# coding=utf-8
import xgboost as xgb
import numpy as np
from load_data import load_data
import argparse
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.metrics import f1_score

def get_train_words(args):
    train_set = {}
    train_data = load_data(args.train_file)
    for line in train_data:
        for word in line[2]:
            train_set[word] = 0
    output = open('train_word.pkl', 'wb')
    pickle.dump(train_set, output)
    
def process(data, args, word2idx):
    title = []
    label = []
    label_dict = {3: 0,5: 1,2319: 2,7: 3,1472: 4,1648: 5,591: 6,571: 7,328: 8,1552: 9}
    for line in data:
        temp = [x for x in line[2] if x in word2idx]
        title.append(' '.join(temp))
        label.append(label_dict[line[5]])
    return title, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Training Data file')
    parser.add_argument('test_file', type=str, help='Testing data file')
    parser.add_argument('word2idx', type=str, help='word to idx file, which has deleted the uncommonly uesd words')
    
    args = parser.parse_args()

    
    f = open(args.word2idx,'rb')
    word2idx = pickle.load(f)

    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()

    # train
    train_data = load_data(args.train_file)
    title, label = process(train_data, args, word2idx)
    del train_data
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(title))
    dtrain = xgb.DMatrix(tfidf, label=label)
    param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':10}  # 参数
    evallist  = [(dtrain,'train')]  # 这步可以不要，用于测试效果
    num_round = 200 # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist)

    # test
    test_data = load_data(args.test_file)
    title, label = process(test_data, args, word2idx)
    del test_data
    tfidf = tfidftransformer.transform(vectorizer.transform(title))
    dtest = xgb.DMatrix(tfidf)  # label可以不要，此处需要是为了测试效果
    
    preds = bst.predict(dtest)

    score = f1_score(label, preds, average='weighted')
    print('acc: ', np.mean(preds == label))
    print('F1 score: ', score)

if __name__ == '__main__':

    main()