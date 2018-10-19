import pandas as pd
from sklearn.metrics import f1_score

true_test_path = 'true_test.csv'
pred_test_path = '../bagging/final_result.txt'

true_test_data = pd.read_csv(true_test_path)
target1 = true_test_data['cate1'].tolist()
target2 = true_test_data['cate2'].tolist()
target3 = true_test_data['cate3'].tolist()

pred1, pred2, pred3 = [], [], []
with open(pred_test_path, 'r') as fp:
    fp.readline()
    for line in fp.readlines():
        line = line.strip().split('\t')
        pred1.append(int(line[1]))
        pred2.append(int(line[2]))
        pred3.append(int(line[3]))

cate1_score = f1_score(target1, pred1, average='macro')
cate2_score = f1_score(target2, pred2, average='macro')
cate3_score = f1_score(target3, pred3, average='macro')
weighted_score= 0.1 * cate1_score + 0.3 * cate2_score + 0.6 * cate3_score
print('cate1 score:', cate1_score)
print('cate2 score:', cate2_score)
print('cate3 score:', cate3_score)
print('weighted score:', weighted_score)