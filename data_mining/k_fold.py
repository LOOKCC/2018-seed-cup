#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse


def split_idx(train_file, test_size, save_path):
    with open(train_file, 'r') as f:
        title = f.readline()
        lines = f.readlines()
        total_data = len(lines)
    x = np.zeros((total_data, 1))
    y = range(total_data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    with open(os.path.join(save_path, "train.txt"), 'w') as f:
        f.write(title)
        for i in y_train:
            f.write(lines[i])

    with open(os.path.join(save_path, "test.txt"), 'w') as f:
        f.write(title)
        for i in y_test:
            f.write(lines[i])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Training Data file')
    parser.add_argument('test_size', type=float, help='test proportion')
    parser.add_argument('save_path', type=str, help='the dir to save')
    args = parser.parse_args()    

    split_idx(args.train_file, args.test_size, args.save_path)
