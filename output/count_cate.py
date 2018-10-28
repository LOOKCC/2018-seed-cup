import numpy as np


def count(label):
    if label in cate_count:
        cate_count[label] += 1
    else:
        cate_count[label] = 1


if __name__ == '__main__':
    cate_count = {}
    with open('submit_ordered.txt', 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            count(line[1])
            count(line[2])
            count(line[3])
    print(cate_count)
