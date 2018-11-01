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
python word2vec.py
python word2idx.py
python cate2idx.py
python data2idx.py
python make_mask.py

cd ..
# train models
python train.py -m 1 --h_d 64 -d --ckpt ./checkpoint/cate1_classifier.pth
python train.py -m 2 --h_d 128 -d --ckpt ./checkpoint/cate2_classifier.pth
python train.py -m 3 --h_d 256 -d --ckpt ./checkpoint/cate3_classifier.pth
# if resuming training
python train.py -m 1 --ckpt ./checkpoint/cate1_classifier.pth -r
python train.py -m 2 --ckpt ./checkpoint/cate2_classifier.pth -r
python train.py -m 3 --ckpt ./checkpoint/cate3_classifier.pth -r
# eval
python eval.py --clf1 ./checkpoint/cate1_classifier.pth --clf2 ./checkpoint/cate2_classifier.pth --clf3 ./checkpoint/cate3_classifier.pth
# test and get submit.txt
python test.py --clf1 ./checkpoint/cate1_classifier.pth --clf2 ./checkpoint/cate2_classifier.pth --clf3 ./checkpoint/cate3_classifier.pth --save_path submit.txt
```

## 常用深度学习模型运行步骤

```shell
# get embedding data as above ...
cd swem/preproc
TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    output
    README.md
```


<br/><br/><br/><br/>

--------------------------------------------------------------------------------


## 三、机器学习主要思路

机器学习方法主要是先对数据进行处理,之后使用诸如贝叶斯,boosting之类的方法进行预测.主要分为数据处理和模型选择,以及参数调优这几个过程.

### 数据的读取&处理

首先读取数据,之后进行简单的处理,将类别转化为标签,然后使用tfidf处理word,因为考虑到汉语的语义和字符的相关性不大,所以只是用了word部分,并且将title和description合在了一起.其中也曾经考虑到样本不均衡的问题,还进行了重新对训练集和验证集的划分.

### 模型选择

这里尝试了不少的模型,其中有SVM,贝叶斯,xgboost,最后经过测试,xgboost的效果是最好的.xgboost是一个在数据挖掘常用的库,使用起来方便,训练起来对硬件的要求也比较低,非常适合此类比赛.

### 优化思路

优化方面的话,主要有几个参数需要调整,首先是树的深度,其次是学习速率,还有就是迭代次数,这些都是需要正在训练中根据训练集和验证集的结果进行调整的.

### 最后测试

经过最后的测是,并没有下面的深度学习的得到的最后的分数高,机器学习在语义方面还是没有理解到位,也就是表现力仍然不够,但是在这方面的花费的时间还是有意义的.

<br/><br/><br/><br/>

--------------------------------------------------------------------------------

## 四、深度学习主要思路

深度学习主要先进行词向量的预训练, 之后采用TextCNN、LSTM、GRU等和基于swem(Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms(ACL 2018))作为特征表示的模型进行训练和参数调优.

### 数据的读取&处理

1. word和char分别进行词向量预训练 (model=cbow, criterion=negative sampling) 

2. 数据集中word, char以及cate id到idx的处理

3. 针对此次比赛数据的特性, 挖掘出cate1->cate2->cate3的关联, 生成mask.pkl与class_info.pkl用于之后对cate预测的筛选

### 模型选择

此次学习目标是进行文本分类, 首先尝试常用的文本分类模型，如TextCNN、LSTM、GRU等. 同时，在初赛尝试naive bayes发现效果并不差, 猜想文本的词序对文本分类的影响不大, 于是尝试采用swem作为特征表示层即仅使用pooling层处理文本词向量, 忽略词序对预测结果的影响.

### 优化思路

- SWEM
  - 表示层： 最初仅使用max pooling效果一般, 于是考虑使用swem论文中的swem cat, 即将max pooling和avg pooling得到的向量进行concat, 效果有所提升, 于是之后的训练都采用swem cat作为表示层. 训练输入数据尝试过仅使用word title, 使用word title和word description, 以及使用训练集的所有word, char的数据. 
  - 分类器: 最初使用分级multi task训练, 即cate1, cate2, cate3共用相同特征表示层，其中loss = 0.1 * cate1_loss + 0.3 * cate2_loss + 0.6 * cate3_loss. 后考虑到数据集特征, 尝试对cate1, cate2, cate3分类器并行训练, 各自拥有自己的特征表示层, 得到了当前最佳结果.
- 其他模型
  - LSTM、GRU、TextCNN在word数据上效果类似，但在char数据上TextCNN更好。同时，TextCNN原论文中采用大小为3、4、5的卷积核，最后将特征相连。但若分别训练三个卷积层，结果相加将得到更好的效果
  - 同时也尝试了dropout、batch normalization、layer normalization、weight decay等，TextCNN在添加 batch normalization 层后效果提升明显。
  - 测试发现，将title与description连接后输入，比单独输入后融合或者只用其中一个效果要好。

### 最后测试

对swem采用swem cat作为表示层, 对三个分类器分别进行训练, 输入数据仅使用训练集的word数据并对每条文本数据使用0.2的几率drop其中的部分单词作为数据增强.  
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
| Byes                   | 0.7586            |0.8849|0.8024|0.7157|
### 深度学习
|         模型         |        总结果         | cate1 | cate2 | cate3 |
|--------------------:|:--------------------:|:------:|:-----:|:------|
|  (CNN+bn+relu+fc)*1 | 0.8540               |0.9545       |0.8884      |0.8201      |
| conv_3+conv_4+conv_5 word| 0.8698          |0.9579       |0.8993      |0.8404      |
|  GRU*1              | 0.8468               |0.9496       |0.8799      |0.8131      |
|  LSTM*1             | 0.8543               |0.9557       |0.8868      |0.8212      |
|3 * concat(maxpooling, avgpooling ) -> fc -> bn -> fc -> softmax | 0.8520| ? | ?　| ? |
|3 * concat(maxpooling, avgpooling ) -> fc -> bn -> fc -> softmax  add char| 0.8579| 0.9557|0.8888|0.8262|

## 六、贡献者

* [LOOKCC](https://github.com/LOOKCC)
* [platoneko](https://github.com/platoneko)
* [Q . J . Y](https://github.com/qjy981010)

From HUST, 我们没有队名

2018.11.01 Finished.

