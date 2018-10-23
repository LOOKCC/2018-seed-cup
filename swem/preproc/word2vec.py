from gensim.models import Word2Vec
import torch

model_path = './word_embedding_512_.model'
save_path = './word2vec.pth'

def make_vec():
    print('==> Saving vectors tensor in {}'.format(save_path))
    model = Word2Vec.load(model_path)
    vectors = torch.tensor(model.wv.vectors)
    padding = torch.zeros((1, vectors.size(1)))
    vectors = torch.cat([padding, vectors])
    torch.save(vectors, save_path)

if __name__ == '__main__':
    make_vec()