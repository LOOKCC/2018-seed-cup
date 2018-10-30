import argparse
import functools
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from load_data import *


def process(data, level=0):
    ids = []
    title = []
    label = []
    for line in data:
        ids.append(line[0])
        title.append(' '.join(line[2]))
        if len(line) > 5:
            label.append(line[level+5])
    return ids, title, label


def predict(train_leveled_data, test_data, classes):
    pipeline = Pipeline([('vect', CountVectorizer()),
                         # ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         # ('clf', SVC(kernel = 'linear')),
                         # ('clf', SGDClassifier(loss='hinge',
                         #                       penalty='l2',
                         #                       alpha=5e-5 * 4.5**len(classes),
                         #                       max_iter=10,
                         #                       shuffle=True,
                         #                       class_weight='balanced',
                         #                       random_state=42)),
                         ])
    data = get_class_data(train_leveled_data, classes)
    title, label = process(data, len(classes))[1:]
    if len(set(label)) > 1:
        pipeline = pipeline.fit(title, label)
        ids, title, label = process(test_data, len(classes))
        result = pipeline.predict(title)
    else:
        ids, title = process(test_data, len(classes))[:2]
        result = [label[0]]*len(ids)
    result = dict(zip(ids, result))
    return result


def list2dict(data, predict_result):
    if isinstance(data, dict):
        return {k: list2dict(data[k], predict_result) for k in data.keys()}
    else:
        leveled_data = {}
        for line in data:
            c = predict_result[line[0]]
            if c not in leveled_data:
                leveled_data[c] = []
            leveled_data[c].append(line)
        return leveled_data


def main(args):
    train_leveled_data = load_leveled_data(args.train_file)[0]
    test_data = load_data(args.test_file)  # not be leveled now
    test_leveled_data = test_data

    final_result = {line[0]: [] for line in test_data}

    classes = [[]]
    for level in range(3):
        start = datetime.now()
        result = {}
        for _ in range(len(classes)):
            c = classes.pop(0)
            level_data = functools.reduce(
                lambda x, y: x[y], [test_leveled_data, *c])
            result.update(predict(train_leveled_data, level_data, c))

            keys = functools.reduce(
                lambda x, y: x[y], [train_leveled_data, *c]).keys()
            classes.extend([c+[key] for key in keys])

        test_leveled_data = list2dict(test_leveled_data, result)

        for item_id in final_result:
            final_result[item_id].append(result[item_id])
        print('level {} over\t\t   time: {}'.format(
            level, datetime.now()-start))

    if len(test_data[0]) > 5:
        scores = []
        weights = [0.1, 0.3, 0.6]
        for level in range(3):
            result, label = [], []
            for line in test_data:
                label.append(line[level+5])
                result.append(final_result[line[0]][level])
            score = f1_score(label, result, average='macro')
            scores.append(score)
            print('level', level, 'acc: ', accuracy_score(result, label))
            print('level', level, 'F1 score: ', score)
        print('weighted score: ', sum(
            [scores[i]*weights[i] for i in range(3)]))

    with open(args.predict_file, 'w') as fp:
        fp.write('item_id\tcate1_id\tcate2_id\tcate3_id\n')
        for item_id in final_result:
            fp.write('\t'.join((item_id, *[str(x)
                                           for x in final_result[item_id]])))
            fp.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Training Data file')
    parser.add_argument('test_file', type=str, help='Testing data file')
    parser.add_argument('predict_file', type=str, help='Data file to output')
    args = parser.parse_args()
    main(args)
