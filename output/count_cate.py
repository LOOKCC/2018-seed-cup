import numpy as np
cate_count = {}
with open('subumit_ordered.txt', 'r') as f:
    f.readline()
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
