from gensim.models import Word2Vec
import pickle
import argparse

word_model_path = './word_embedding_512_.model'
word_save_path = './word2idx.pth'
char_model_path = './char_embedding_256_.model'
char_save_path = './char2idx.pth'


def make_idx(save_path, model_path):
    print('==> Saving indices table in {}'.format(save_path))
    model = Word2Vec.load(model_path)
    indices = {val: idx+1 for idx, val in enumerate(model.wv.index2word)}
    with open(save_path, 'wb') as fp:
        pickle.dump(indices, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=1,
                        help='1 for word, 0 for char; default=1')
    args = parser.parse_args()
    if args.w:
        model_path = word_model_path
        save_path = word_save_path
    else:
        model_path = char_model_path
        save_path = char_save_path
    make_idx(save_path, model_path)