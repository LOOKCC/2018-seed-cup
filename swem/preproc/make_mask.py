import pickle

CATE1_CNT = 20
CATE2_CNT = 135
CATE3_CNT = 265

save_path = './mask.pkl'

with open('./train_cate.pkl', 'rb') as fp:
    labels = pickle.load(fp)

mask1 = [set() for i in range(CATE1_CNT)]
mask2 = [set() for i in range(CATE2_CNT)]

for (l1, l2, l3) in labels:
    mask1[l1].add(l2)
    mask2[l2].add(l3)

ca2_set = set(range(CATE2_CNT))
ca3_set = set(range(CATE3_CNT))
for i in range(CATE1_CNT):
    mask1[i] = list(ca2_set - mask1[i])
for i in range(CATE2_CNT):
    mask2[i] = list(ca3_set - mask2[i])
print('Example mask1: {}'.format(mask1[0]))
print('Example mask2: {}'.format(mask2[0]))

mask = (mask1, mask2)
print('==> Saving mask to {}'.format(save_path))
with open(save_path, 'wb') as fp:
    pickle.dump(mask, fp)
