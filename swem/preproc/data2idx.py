import pickle

train_b_path = '../../data/train_b.txt'
valid_b_path = '../../data/valid_b.txt'
test_path = '../../data/test_b.txt'
train_a_path = '../../data/train_a.txt'
valid_a_path = '../../data/valid_a.txt'

with open('./word2idx.pkl', 'rb') as fp:
    word2idx = pickle.load(fp)

with open('./preproc/char2idx.pkl', 'rb') as fp:
    char2idx = pickle.load(fp)

with open('cate2idx.pkl', 'rb') as fp:
    ca2idx = pickle.load(fp)


# 0:ch_title, 1: word_title, 2: ch_descrip, 3: word_descrip
def feature2idx(load_path, save_path):
    features = []
    if isinstance(load_path, tuple) or isinstance(load_path, list):
        for path in load_path:
            _word_feature2idx(path, features)
    else:
        _word_feature2idx(load_path, features)
    print('==> Saving preprocessed feature in {}'.format(save_path))
    with open(save_path, 'wb') as fp:
        pickle.dump(features, fp)

def _word_feature2idx(path, features):
    print('==> Transforming words in {} to indices..'.format(path))
    with open(path, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            feature = []
            line = line.strip().split('\t')
            feature.append([word2idx[word] for word in line[2].split(',') if word in word2idx])
            feature.append([word2idx[word] for word in line[4].split(',') if word in word2idx])
            features.append(feature)


def cate2idx(load_path, save_path):
    cate = []
    if isinstance(load_path, tuple) or isinstance(load_path, list):
        for path in load_path:
            _cate2idx(path, cate)
    else:
        _cate2idx(load_path, cate)
    print('==> Saving preprocessed cate in {}'.format(save_path))
    with open(save_path, 'wb') as fp:
        pickle.dump(cate, fp)

def _cate2idx(path, cate):
    print('==> Transforming cate in {} to indices..'.format(path))
    with open(path, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            ca = []
            line = line.strip().split('\t')
            ca.append(ca2idx[0][line[5]])
            ca.append(ca2idx[1][line[6]])
            ca.append(ca2idx[2][line[7]])
            cate.append(ca)

if __name__ == '__main__':
    feature2idx((train_b_path, train_a_path, valid_a_path), './train_words.pkl')
    feature2idx(valid_b_path, './valid_words.pkl')
    feature2idx(test_path, './test_words.pkl')

    cate2idx((train_b_path, train_a_path, valid_a_path), './train_cate.pkl')
    cate2idx(valid_b_path, './valid_cate.pkl')