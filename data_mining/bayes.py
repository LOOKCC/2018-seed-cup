import argparse
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from load_data import *


def process(data, args):
    ids = []
    title = []
    label = []
    for line in data:
        ids.append(line[0])
        title.append(' '.join(line[2]))
        label.append(line[5])
    return ids, title, label


def predict(train_leveled_data, test_leveled_data, classes):
    data = get_class_data(train_leveled_data, classes)
    title, label = process(data)[1:]
    pipeline = pipeline.fit(title, label)
    data = get_class_data(test_leveled_data, classes)
    ids, title, label = process(data)
    result = pipeline.predict(title)
    result = dict(zip(ids, result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Training Data file')
    parser.add_argument('test_file', type=str, help='Testing data file')
    args = parser.parse_args()

    pipeline = Pipeline([('vect', CountVectorizer()),
                         # ('tfidf', TfidfTransformer()),
                         # ('clf', MultinomialNB()),
                         # ('clf', SVC(kernel = 'linear')),
                         ('clf', SGDClassifier(loss='hinge',
                                            penalty='l2',
                                            alpha=1e-5,
                                            n_iter=10,
                                            random_state=42)),
    ])

    start = datetime.now()
    train_leveled_data = load_leveled_data(args.train_file)[0]
    test_leveled_data, item_ids = load_leveled_data(args.test_file)
    final_result = {item_id: [] for item_id in item_ids}

    classes = []
    for level in range(3):
        result = predict(train_leveled_data, test_leveled_data, classes)

    print('training over ', datetime.now()-start)

    score = f1_score(label, result, average='weighted')
    print('acc: ', np.mean(result == label))
    print('F1 score: ', score)

if __name__ == '__main__':
    main()
