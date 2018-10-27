from gensim.models import Word2Vec
import torch
import argparse

word_model_path = './word_embedding_512_.model'
word_save_path = './word2vec.pth'
char_model_path = './char_embedding_256_.model'
char_save_path = './char2vec.pth'

def make_vec(save_path, model_path):
    print('==> Saving vectors tensor in {}'.format(save_path))
    model = Word2Vec.load(model_path)
    vectors = torch.tensor(model.wv.vectors)
    padding = torch.zeros((1, vectors.size(1)))
    vectors = torch.cat([padding, vectors])
    torch.save(vectors, save_path)

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
    make_vec(save_path, model_path)