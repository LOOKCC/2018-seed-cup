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
    title = []
    label = []
    for line in data:
        title.append(' '.join(line[2]))
        label.append(line[5])
    return title, label


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
    train_data = load_data(args.train_file)
    title, label = process(train_data, args)
    del train_data
    pipeline = pipeline.fit(title, label)
    print('training over ', datetime.now()-start)

    start = datetime.now()
    test_data = load_data(args.test_file)
    title, label = process(test_data, args)
    del test_data
    result = pipeline.predict(title)
    print('testing over ', datetime.now()-start)

    score = f1_score(label, result, average='weighted')
    print('acc: ', np.mean(result == label))
    print('F1 score: ', score)

if __name__ == '__main__':
    main()
