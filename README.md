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

```

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
            train.txt
            test.txt
    bagging
        bagging.py
    data_mining
        bayes.py
        k_fold.py
        load_data.py
        mining.py
        xgb.py
    utils
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


### 数据的读取&处理


### 模型选择


### 优化思路

### 最后测试

<br/><br/><br/><br/>

--------------------------------------------------------------------------------

## 五、贡献者

* [LOOKCC](https://github.com/LOOKCC)
* [platoneko](https://github.com/platoneko)
* [Q . J . Y](https://github.com/qjy981010)

From HUST, 我们没有队名

2018.10.22 Finished.

