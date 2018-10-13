import pickle
import argparse
from load_data import load_data
from collections import Counter


def get_class_info(data):
    class_info = {}
    for line in data:
        level1, level2, level3 = line[5:8]
        if level1 not in class_info:
            class_info[level1] = {}
        if level2 not in class_info[level1]:
            class_info[level1][level2] = {}
        if level3 not in class_info[level1][level2]:
            class_info[level1][level2][level3] = 0
        class_info[level1][level2][level3] += 1
    with open('../data/class_info.pkl', 'wb') as fp:
        pickle.dump(class_info, fp)


def word_count(data):
    counter = Counter()
    for line in data:
        counter.update(line[2])
        counter.update(line[4])
    with open('word_count.csv', 'w') as fp:
        for x,y in counter.items():
            fp.write(x + ',' + str(y) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Data file')
    args = parser.parse_args()
    
    data = load_data(args.file)
    class_info = get_class_info(data)
    word_count(data)
