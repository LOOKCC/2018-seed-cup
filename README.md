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
* python-pytorch-suda
* python-torchtext
* python-numpy
* python-gensim
* python-scikit-learn

### 运行步骤

## xgboost运行步骤
```
cd data_mining
# for train and valid
python xgb.py --train_file=../data/train_a.txt --test_file=../data/valid_a.txt --class_info=../data/class_info.pkl
# for train and test, the ordered result is in submit_ordered.txt
python xgb.py --train_file=../data/train_a.txt --test_file=../data/test_a.txt --class_info=../data/class_info.pkl --test
```

## swem运行步骤
```
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

<br><br><br>

--------------------------------------------------------------------------------


## 二、项目架构


```
    data
        class_info.pkl
        label2idx.pkl
        word2idx.pkl
        test_a.txt
        train_a.txt
        valid_a.txt
        k_fold
    bagging
        bagging.py
    data_mining
        bayes.py
        k_fold.py
        load_data.py
        mining.py
        xgb.py
    swem
        eval.py
        test.py
        train.py
        utils
            dataset.py
        preproc
            cate2idx.py
            data2idx.py
            make_mask.py
            word2idx.py
            word2vec.py
            word_embedding.py
        models
            cate1_classifier.py
            cate2_classifier.py
            cate3_classifier.py
            represent_layer.py
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

深度学习主要先进行词向量的预训练, 之后采用基于swem(Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms(ACL 2018))作为特征表示的模型进行训练和参数调优.

### 数据的读取&处理

1. word和char分别进行词向量预训练 (model=cbow, criterion=negative sampling) 

2. 数据集中word, char以及cate id到idx的处理

3. 针对此次比赛数据的特性, 挖掘出cate1->cate2->cate3的关联, 生成mask.pkl用于之后对cate预测的筛选

### 模型选择

此次学习目标是进行文本分类, 在进行最初的naive bayes后发现效果并不差, 猜想文本的词序对文本分类的影响不大, 于是尝试采用swem作为特征表示层即仅使用pooling层处理文本词向量, 忽略词序对预测结果的影响.

### 优化思路

- 表示层： 最初仅使用max pooling效果一般, 于是考虑使用swem论文中的swem cat, 即将max pooling和avg pooling得到的向量进行concat, 效果有所提升, 于是之后的训练都采用swem cat作为表示层. 训练输入数据尝试过仅使用word title, 使用word title和word desciption, 以及使用训练集的所有word, char的数据. 

- 分类器: 最初使用分级multi task训练, 即cate1, cate2, cate3共用相同特征表示层，其中loss = 0.1 * cate1_loss + 0.3 * cate2_loss + 0.6 * cate3_loss. 后考虑到数据集特征, 尝试对cate1, cate2, cate3分类器并行训练, 各自拥有自己的特征表示层, 得到了当前最佳结果.

### 最后测试

最终测试采用swem cat作为表示层, 对三个分类器分别进行训练, 输入数据仅使用训练集的word数据并对每条文本数据使用0.2的几率drop其中的部分单词作为数据增强.

<br/><br/><br/><br/>

--------------------------------------------------------------------------------

## 五、贡献者

* [LOOKCC](https://github.com/LOOKCC)
* [platoneko](https://github.com/platoneko)
* [Q . J . Y](https://github.com/qjy981010)

From HUST, 我们没有队名

2018.10.22 Finished.

