from gensim.models import Word2Vec
import pickle

model_path = './word_embedding_512_.model'
save_path = './word2idx.pkl'

def make_idx():
    print('==> Saving words indices table in {}'.format(save_path))
    model = Word2Vec.load(model_path)
    indices = {val: idx+1 for idx, val in enumerate(model.wv.index2word)}
    with open(save_path, 'wb') as fp:
        pickle.dump(indices, fp)

if __name__ == '__main__':
    make_idx()