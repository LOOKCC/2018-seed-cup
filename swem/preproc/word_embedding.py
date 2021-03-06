from gensim.models import Word2Vec
import argparse
from utils.timer import timer

train_a_path = '../../data/train_a.txt'
valid_a_path = '../../data/valid_a.txt'
test_a_path = '../../data/test_a.txt'
train_b_path = '../../data/train_b.txt'
valid_b_path = '../../data/valid_b.txt'
test_b_path = '../../data/test_b.txt'


# 0: id, 1:ch_title, 2: word_title, 3: ch_descrip, 4: word_descrip
def load_words(path):
    print('==> Loading words from', path)
    word_sentences = []
    with open(path, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            line = line.strip().split('\t')
            word_sentences.append(line[2].split(','))
            word_sentences.append(line[4].split(','))
    print('Word sentence example:', word_sentences[0])
    return word_sentences


def load_chars(path):
    print('==> Loading chars from', path)
    char_sentences = []
    with open(path, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            line = line.strip().split('\t')
            char_sentences.append(line[1].split(','))
            char_sentences.append(line[3].split(','))
    print('Chars sentence example:', char_sentences[0])
    return char_sentences


@timer
def word_embedding(sentences, model_save_path, txt_save_path, args):
    print('==> Building embedding model..')
    model = Word2Vec(sentences, size=args.size, window=args.window, sg=args.sg, hs=args.hs, negative=args.negative,
                     cbow_mean=args.cbow_mean, min_count=args.min_count, iter=args.iter, workers=4)
    print('==> Saving model in {}'.format(model_save_path))
    model.save(model_save_path)
    print('Finished saving model')
    print('==> Saving txt in {}'.format(txt_save_path))
    model.wv.save_word2vec_format(txt_save_path)
    print('Finished saving txt')


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word', type=int, default=1,
                        help='1 to embed words, 0 to embed chars; default=1')
    parser.add_argument('--size', type=int, default=512,
                        help='embedding size; default=512')
    parser.add_argument('--window', type=int, default=5,
                        help='window size; default=5')
    parser.add_argument('--sg', type=int, default=0,
                        help='0 for CBOW, 1 for SkipGram; default=0')
    parser.add_argument('--hs', type=int, default=0,
                        help='0 for Negative Sampling, 1 for Hierarchical Softmax; default=0')
    parser.add_argument('--negative', type=int, default=10,
                        help='if using Negative Sampling, it will be the number of negative samples; default=10')
    parser.add_argument('--cbow_mean', type=int, default=1,
                        help='if using CBOW, 1 to use mean of surrounding words\' vectors, '
                        '0 to use sum of surrounding words\' vectors; default=1')
    parser.add_argument('--min_count', type=int, default=5,
                        help='lower limit of words\' frequency; default=5')
    parser.add_argument('--iter', type = int, default = 10,
                        help = 'iter times; default=10')
    args=parser.parse_args()
    return args


if __name__ == '__main__':
    args=parse_cmd()
    sentences=[]
    if args.word:
        sentences.extend(load_words(train_a_path))
        sentences.extend(load_words(valid_a_path))
        sentences.extend(load_words(test_a_path))
        sentences.extend(load_words(train_b_path))
        sentences.extend(load_words(valid_b_path))
        sentences.extend(load_words(test_b_path))
        word_embedding(sentences, './word_embedding_' + str(args.size) + '_.model',
                       './word_embedding_' + str(args.size) + '_.txt', args)
    else:
        sentences.extend(load_chars(train_a_path))
        sentences.extend(load_chars(valid_a_path))
        sentences.extend(load_chars(test_a_path))
        sentences.extend(load_chars(train_b_path))
        sentences.extend(load_chars(valid_b_path))
        sentences.extend(load_chars(test_b_path))
        word_embedding(sentences, './char_embedding_' + str(args.size) + '_.model',
                       './char_embedding_' + str(args.size) + '_.txt', args)
