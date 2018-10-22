import pickle

train_path = '../../data/train_a.txt'
valid_path = '../../data/valid_a.txt'
test_path = '../../data/test_a.txt'

with open('./word2idx.pkl', 'rb') as fp:
    word2idx = pickle.load(fp)

# with open('./preproc/char2idx.pkl', 'rb') as fp:
    # char2idx = pickle.load(fp)

# 0:ch_title, 1: word_title, 2: ch_descrip, 3: word_descrip
def feature2idx(load_path, save_path):
    print('==> Transforming words in {} to indices..'.format(load_path))
    with open(load_path, 'r') as fp:
        fp.readline()
        features = []
        for line in fp.readlines():
            feature = []
            line = line.strip().split('\t')
            # feature.append([char2idx[char] for char in line[1].split(',') if char in char2idx])
            feature.append([word2idx[word] for word in line[2].split(',') if word in word2idx])
            # feature.append([char2idx[char] for char in line[3].split(',') if char in char2idx])
            feature.append([word2idx[word] for word in line[4].split(',') if word in word2idx])
            features.append(feature)
    print('==> Saving preprocessed feature in {}'.format(save_path))
    with open(save_path, 'wb') as fp:
        pickle.dump(features, fp)



with open('cate2idx.pkl', 'rb') as fp:
    _cate2idx = pickle.load(fp)

def cate2idx(load_path, save_path):
    print('==> Transforming cate in {} to indices..'.format(load_path))
    with open(load_path, 'r') as fp:
        fp.readline()
        cate = []
        for line in fp.readlines():
            ca = []
            line = line.strip().split('\t')
            ca.append(_cate2idx[0][line[5]])
            ca.append(_cate2idx[1][line[6]])
            ca.append(_cate2idx[2][line[7]])
            cate.append(ca)
    print('==> Saving preprocessed cate in {}'.format(save_path))
    with open(save_path, 'wb') as fp:
        pickle.dump(cate, fp)


if __name__ == '__main__':
    feature2idx(train_path, './train_words.pkl')
    feature2idx(valid_path, './valid_words.pkl')
    feature2idx(test_path, './test_words.pkl')

    cate2idx(train_path, './train_cate.pkl')
    cate2idx(valid_path, './valid_cate.pkl')