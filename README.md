# Commodity Category Prediction

<p align="center"> by Q . J . Y, LOOKCC, platoneko </p>

<br/><br/>

--------------------------------------------------------------------------------
## 一、运行环境

### 操作系统和语言

* OS: Ubuntu 16.04.5 LTS
* Python: 3.7.0

### 依赖库

* python-xgboost
* python-pytorch-cuda
* python-torchtext
* python-numpy
* python-gensim
* python-scikit-learn

### 运行步骤

## xgboost运行步骤

```shell
cd data_mining
# Please put the test, valid, train data in ../data, and the following will generate a file named class_info.pkl in ../data for training
python mining.py ../data/train_b.txt
# train and valid with char
python xgb_save.py --save_path=./models_char --epoch=300 --use_char
# train and valid with word
python xgb_save.py --save_path=./models_word --epoch=300 --use_word
# for test, the ordered result is in submit_ordered.txt, here take the word for example
python xgb_save.py --test_file=../data/test_b.txt --save_path=./models_word --test --use_word
```

## swem运行步骤

```shell
cd swem/preproc
# preproc data
python word_embedding.py -w 1
python word_embedding.py -w 0
python word2vec.py -w 1
python word2vec.py -w 0
python word2idx.py -w 1
python word2idx.py -w 0
python cate2idx.py
python data2idx.py -w 1 --cate
python data2idx.py -w 0
python make_mask.py

cd ..
# train models
# -m for cate n(1, 2, 3)
# -w for word(1) or char(0)
# --hier for swemhier(1) or swemcat(0)
python train.py -m 1 -w 1 --h_d 64 --embedding_dim 512 --ckpt w_cate1.pth
python train.py -m 2 -w 1 --h_d 256 --embedding_dim 512 --ckpt w_cate2.pth
python train.py -m 3 -w 1 --h_d 512 --embedding_dim 512 --ckpt w_cate3.pth
python train.py -m 1 -w 0 --h_d 64 --embedding_dim 256 --ckpt c_cate1.pth
python train.py -m 2 -w 0 --h_d 256 --embedding_dim 256 --ckpt c_cate2.pth
python train.py -m 3 -w 0 --h_d 512 --embedding_dim 256 --ckpt c_cate3.pth
python train.py -m 1 -w 1 --h_d 64 --hier 1 --embedding_dim 512 --ckpt hier_w_cate1.pth
python train.py -m 2 -w 1 --h_d 256 --hier 1 --embedding_dim 512 --ckpt hier_w_cate2.pth
python train.py -m 3 -w 1 --h_d 512 --hier 1 --embedding_dim 512 --ckpt hier_w_cate3.pth
python train.py -m 1 -w 0 --h_d 64 --hier 1 --embedding_dim 256 --ckpt hier_c_cate1.pth
python train.py -m 2 -w 0 --h_d 256 --hier 1 --embedding_dim 256 --ckpt hier_c_cate2.pth
python train.py -m 3 -w 0 --h_d 512 --hier 1 --embedding_dim 256 --ckpt hier_c_cate3.pth
# if resuming training
python train.py -m 1 -w 1 --ckpt w_cate1.pth -r
python train.py -m 2 -w 1 --ckpt w_cate2.pth -r
python train.py -m 3 -w 1 --ckpt w_cate3.pth -r
python train.py -m 1 -w 0 --ckpt c_cate1.pth -r
python train.py -m 2 -w 0 --ckpt c_cate2.pth -r
python train.py -m 3 -w 0 --ckpt c_cate3.pth -r
python train.py -m 1 -w 1 --ckpt hier_w_cate1.pth -r
python train.py -m 2 -w 1 --ckpt hier_w_cate2.pth -r
python train.py -m 3 -w 1 --ckpt hier_w_cate3.pth -r
python train.py -m 1 -w 0 --ckpt hier_c_cate1.pth -r
python train.py -m 2 -w 0 --ckpt hier_c_cate2.pth -r
python train.py -m 3 -w 0 --ckpt hier_c_cate3.pth -r
# eval
python eval.py
# test and get submit.txt
python test.py --save_path ../output/swem_wordchar_out.txt
```
## glu运行步骤

```shell
cd glu
# train models
# -m for cate n(1, 2, 3)
python train.py -m 1 --h_d 64 --embedding_dim 512 --ckpt glu_w_cate1.pth
python train.py -m 2 --h_d 256 --embedding_dim 512 --ckpt glu_w_cate2.pth
python train.py -m 3 --h_d 512 --embedding_dim 512 --ckpt glu_w_cate3.pth
# if resuming training
python train.py -m 1 --ckpt glu_w_cate1.pth -r
python train.py -m 2 --ckpt glu_w_cate2.pth -r
python train.py -m 3 --ckpt glu_w_cate3.pth -r

# eval
python eval.py
```

## 常用深度学习模型运行步骤

```shell
# get embedding data as above ...
cd swem/preproc
python word_embedding.py -w 1 --size=256
python word_embedding.py -w 0 --size=256

cd deep_learning
mkdir embedding
mv ../swem/preproc/embedding_256.txt embedding
vim models/model.py  # cnn or lstm/gru
# train and valid
python main.py --device=cuda:0 --merge
# eval
python main.py --device=cuda:0 --merge --test --snapshot=snapshot/model_{number}.pth
```

<br><br><br>

--------------------------------------------------------------------------------


## 二、项目架构


```
    data
        class_info.pkl
        label2idx.pkl
        word2idx.pkl
        test_b.txt
        train_b.txt
        valid_b.txt
    bagging
        bagging.py
    data_mining
        bayes.py
        svm.py
        load_data.py
        mining.py
        xgb_save.py
        models
            *.pkl
    deep_learning
        embedding
            embedding_256.txt
            vec_cache
        models
            model.py
            cnn.py
            conv.py
            gru.py
            lstm.py
            rcnn.py
            rescnn.py
            res_lstm.py
            textcnn.py
        snapshot
            model_{epoch}.pth
        main.py
        load_data.py
        swem
        models
            __init__.py
            represent_layer.py
            cate1_classifier.py
            cate2_classifier.py
            cate3_classifier.py
        preproc
            word_embedding.py
            word2idx.py
            word2vec.py
            cate2idx.py
            data2idx.py
            make_mask.py
        utils
            __init__.py
            dataset.py
            timer.py
            class_weight.py
        checkpoint
            *.pth
        train.py
        eval.py
        test.py
    glu
        models
            __init__.py
            glu.py
            cate1_classifier.py
            cate2_classifier.py
            cate3_classifier.py
        utils
            __init__.py
            dataset.py
            timer.py
            class_weight.py
        checkpoint
            *.pth
        train.py
        eval.py
    output
    README.md
```


<br/><br/><br/><br/>

--------------------------------------------------------------------------------


## 三、机器学习主要思路

机器学习方法主要是先对数据进行处理,之后使用诸如贝叶斯,boosting之类的方法进行预测.主要分为数据处理和模型选择,以及参数调优这几个过程.

### 数据的读取&处理

这里需要考虑的是使用什么数据,因为需要将title和description连在一起,经过测试tfidf的效果并不好,所以并没有使用tfidf,之后又测试了单独使用word,char和合起来使用,经过最后的测试,发现合起来效果最好.并且这里利用一二三类的包含关系,训练模型的时候,只训练当前类别下面包含的类别的label,加速训练,减小模型.

### 模型选择

这里尝试了不少的模型,其中有SVM,Bayes,XGBoost,最后经过测试,XGBoost的效果是最好的.XGBoost是一个在数据挖掘常用的库,使用起来方便,训练起来对硬件的要求也比较低,非常适合此类比赛.

### 优化思路

优化方面的话,主要有几个参数需要调整,首先是树的深度,其次是学习速率,还有就是迭代次数,这里充分利用XGBoost的优点,当其在验证集上的得分不再增加的时候,就停止训练.

### 最后测试

经过最后的测是,经过充分的训练,在的分上已经超过了很多的机器学习模型,但在面对CNN这种提取与语义和特征更强的方法面前,XGBoost的表现力仍然不够,但是在这方面的花费的时间还是有意义的.

<br/><br/><br/><br/>

--------------------------------------------------------------------------------

## 四、深度学习主要思路

深度学习主要先进行词向量的预训练, 之后采用TextCNN、LSTM、GRU、GLU(Gated Linear Unit)等和基于swem(Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms(ACL 2018))作为特征表示的模型进行训练和参数调优.

### 数据的读取&处理

1. word和char分别进行词向量预训练 (model=cbow, criterion=negative sampling) 

2. 数据集中word, char以及cate id到idx的处理

3. 针对此次比赛数据的特性, 挖掘出cate1->cate2->cate3的关联, 生成mask.pkl与class_info.pkl用于之后对cate预测的筛选

### 模型选择

此次学习目标是进行文本分类, 首先尝试常用的文本分类模型，如TextCNN、LSTM、GRU等. 同时，在初赛尝试naive bayes发现效果并不差, 猜想文本的词序对文本分类的影响不大, 于是尝试采用swem作为特征表示层即仅使用pooling层处理文本词向量, 忽略词序对预测结果的影响. 之后还使用了swem hier, glu等较新的特征提取方法作为表示层.

### 优化思路

- SWEM
  - 表示层：尝试了swemcat(maxpooling+avgpooling), swemhier(k=3, 5的avgpooling之后进行maxpooling)以及将两者结合的模型, 该族模型在char数据上表现尚可, 故使用了word和char的全部数据进行训练和预测.
  - 分类器: 考虑到数据集特征, 尝试对cate1, cate2, cate3分类器并行训练, 各自拥有自己的特征表示层, 得到了该模型最佳结果.
- 其他模型
  - LSTM、GRU、TextCNN在word数据上效果类似，但在char数据上TextCNN更好。同时，TextCNN原论文中采用大小为3、4、5的卷积核，最后将特征相连。但若分别训练三个卷积层，结果相加将得到更好的效果
  - 同时也尝试了dropout、batch normalization、layer normalization、weight decay等，TextCNN在添加 batch normalization 层后效果提升明显。
  - 测试发现，将title与description连接后输入，比单独输入后融合或者只用其中一个效果要好。
- GLU
  - 尝试了单独k=3的卷积层, 2个k=3的卷积层通过residual堆叠和concat(k=3, k=5)的类inception模型, 对三个标签分别训练, 在residual和inception上的结果但也只和TextCNN相当, 但模型复杂度较高, 没有进一步改进. 

### 最后测试

对swem采用swemcat和swemhier作为表示层, 对三个分类器分别进行训练, 输入数据使用训练集的全部数据(word char).分类器在隐藏层输出后经过bn层再relu激活.  
对lstm与gru只使用word数据，cnn使用全部数据并将3、4、5大小的卷积层输出结果相加。


<br/><br/><br/><br/>

--------------------------------------------------------------------------------
## 五、结果对比
### 机器学习
|         模型         |        总结果         | cate1 | cate2 | cate3 |
|--------------------:|:--------------------:|:------:|:-----:|:------|
| XGBoost weight word    | 0.8545            |0.9520|0.8878|0.8216|
| XGBoost weight char    | 0.8604           |0.9533|0.8919|0.8291|
| XGBoost                | 0.8516            |0.9472|0.8850|0.8190|
| XGBoost weight word char| 0.8666            |0.9573|0.8977|0.8360|
| SVM                    | 0.8101            |0.9110|0.8485|0.7741|
| Bayes                   | 0.7586            |0.8849|0.8024|0.7157|
## 深度学习
|         模型         |        总结果         | cate1 | cate2 | cate3 |
|--------------------:|:--------------------:|:------:|:-----:|:------|
|  (CNN+bn+relu+fc)*1 | 0.8540               |0.9545       |0.8884      |0.8201      |
| conv_3+conv_4+conv_5 word| 0.8698          |0.9579       |0.8993      |0.8404      |
| conv_3+conv_4+conv_5 word char| 0.8730     |0.9584       |0.9019      |0.8443      |
| RCNN                | 0.8416               |0.9504       |0.8761      |0.8063      |
|  GRU*1              | 0.8468               |0.9496       |0.8799      |0.8131      |
|  LSTM*1             | 0.8543               |0.9557       |0.8868      |0.8212      |
|3 * swemcat word     | 0.8520               | ?           | ?　        | ?          |
|3 * swemcat word char| 0.8579               |0.9557       |0.8888      |0.8262      |
|3*swemcat with swemhier word char| 0.8647   |0.9582       |0.8952      |0.8339      |
|inception(glu_3+glu_5) word| 0.8545         |0.9549       |0.8870       |0.8216      |

## 六、贡献者

* [LOOKCC](https://github.com/LOOKCC)
* [platoneko](https://github.com/platoneko)
* [Q . J . Y](https://github.com/qjy981010)

From HUST, 我们没有队名

2018.11.01 Finished.

