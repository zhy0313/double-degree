# -*- coding: utf-8 -*-
# 好评good.txt(1列)和坏评bad.txt(1列),停用词stop.txt(1列)
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression
from random import shuffle
from nltk.metrics.scores import precision

from nltk.metrics.scores import recall

from nltk.metrics.scores import f_measure
# 获取文本字符串
import io
import math
import pandas as pd
import numpy as np
import nltk
import jieba
from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from nltk.metrics import BigramAssocMeasures
import collections

basePath = '../data/'
def text():
    f1 = io.open(basePath+ 'good-comment.txt', 'r', encoding='utf-8')

    f2 = io.open(basePath + 'bad-comment.txt', 'r', encoding='utf-8')

    line1 = f1.readline()

    line2 = f2.readline()

    str = ''

    while line1:
        str += line1

        line1 = f1.readline()

    while line2:
        str += line2

        line2 = f2.readline()

    f1.close()

    f2.close()
    return str


# 把单个词作为特征

def bag_of_words(words):
    return dict([(word, True) for word in words])


# 把双个词作为特征--使用卡方统计的方法，选择排名前1000的双词

def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式

    bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前1000的双词

    newBigrams = [u + v for (u, v) in bigrams]

    return bag_of_words(newBigrams)

# 把单个词和双个词一起作为特征

def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)

    bigrams = bigram_finder.nbest(score_fn, n)

    newBigrams = [u + v for (u, v) in bigrams]

    a = bag_of_words(words)

    b = bag_of_words(newBigrams)

    a.update(b)  # 把字典b合并到字典a中

    return a  # 所有单个词和双个词一起作为特征


# 返回分词列表如：[['我','爱','北京','天安门'],['你','好'],['hello']]，一条评论一个
def cut_words(lines):
    stop = [line.strip() for line in io.open(basePath + 'stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
    str = []
    for line in lines:
        s = line.split('\n')
        fenci = jieba.cut(s[0], cut_all=False)  # False默认值：精准模式
        str.append(list(set(fenci) - set(stop)))
    return str

# 读取文件并分词
def read_cut_comments(filename, sep='\n', header=None, names=['comment'], key='comment'):
    data = pd.read_csv(basePath + filename, sep=sep, header=header,names=names)
    return cut_words(data[key])






# 获取信息量最高(前number个)的特征(卡方统计)
def chi_features(number):
    posWords = []

    negWords = []

    for items in read_cut_comments(basePath+ 'good-comment.txt'):  # 把集合的集合变成集合
        for item in items:
            posWords.append(item)

    for items in read_cut_comments(basePath+ 'bad-comment.txt'):

        for item in items:
            negWords.append(item)

    word_fd = FreqDist()  # 可统计所有词的词频

    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频

    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1

    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量

    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量

    total_word_count = pos_word_count + neg_word_count

    word_scores = {}  # 包括了每个词和这个词的信息量

    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量

        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理

        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的

    best_words = set([w for w, s in best_vals])

    return dict([(word, True) for word in best_words])


def transform_data(original_data, features):
    transformed_data = []
    for items in original_data:
        a = {}
        for item in items:
            if item in features.keys():
                a[item] = 'True'
        if len(a) == 0 and len(items) != 0:
            a[items[(0+len(items))/2]] = 'True'
            # print items[(0+len(items))/2]
        if len(a) != 0:
            transformed_data.append(a)
    return transformed_data


def build_features(informative_features):
    posFeatures = []
    pos_datas = transform_data(read_cut_comments(basePath + 'good-comment.txt'),informative_features)
    for item in pos_datas:
        posWords = [item, 'pos']
        posFeatures.append(posWords)

    negFeatures = []
    neg_datas = transform_data(read_cut_comments(basePath + 'bad-comment.txt'),informative_features)
    for item in neg_datas:
        negWords = [item, 'neg']
        negFeatures.append(negWords)
    return posFeatures, negFeatures


# 构建训练需要的数据格式：

# [[{'买': 'True', '京东': 'True', '物流': 'True', '包装': 'True', '\n': 'True', '很快': 'True', '不错': 'True', '酒': 'True', '正品': 'True', '感觉': 'True'},  'pos'],

# [{'买': 'True', '\n':  'True', '葡萄酒': 'True', '活动': 'True', '澳洲': 'True'}, 'pos'],

# [{'\n': 'True', '价格': 'True'}, 'pos']]







def score(classifier,trainFeatures, testFeatures):

    testX, test_y = zip(*testFeatures)  # 分离测试集合的数据和标签，便于验证和测试,解压操作

    classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口

    classifier.train(trainFeatures)  # 训练分类器

    test_pred = classifier.classify_many(testX)  # 对测试集的数据进行分类，给出预测的标签

    n = 0

    s = len(test_pred)
    errors = []
    errors_y = []
    for i in range(0, s):
        if test_pred[i] == test_y[i]:
            n = n + 1
        else:
            errors.append(testX[i])
            errors_y.append(test_y[i])
    # print pd.DataFrame({"errorsX":errors,'errorsY':errors_y});
    return  1.0 * n / s    # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

def acc_precision_recall_f1_score(classifier, trainFeatures, testFeatures):
    from nltk.classify import NaiveBayesClassifier
    # classifier = NaiveBayesClassifier.train(trainFeatures)
    classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
    classifier.train(trainFeatures)  # 训练分类器

    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    # print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    # classifier.show_most_informative_features(40)
    return nltk.classify.util.accuracy(classifier, testFeatures),precision(referenceSets['pos'], testSets['pos']),\
           recall(referenceSets['pos'], testSets['pos']),precision(referenceSets['neg'], testSets['neg']),recall(referenceSets['neg'], testSets['neg']), \
           f_measure(referenceSets['pos'], testSets['pos']), f_measure(referenceSets['neg'], testSets['neg']),classifier


def compare_model(models, trainFeatures, testFeatures):
    best_acc = 0
    best_classifier = None
    best_name = ""
    for name, model in models.items():
        accuracy, pos_precision, pos_recall, neg_precision, neg_recall,pos_f1score,neg_f1score, classifier = acc_precision_recall_f1_score(model,trainFeatures,testFeatures)
        print name,":", "accuracy:",accuracy, ",pos precision:",pos_precision,",pos recall:",pos_recall, ",pos f1_score:",pos_f1score, \
            ',neg precision:',neg_precision, ',neg recall:',neg_recall,",neg f1_score:",neg_f1score
        if accuracy > best_acc:
            best_acc = accuracy
            best_classifier = classifier
            best_name = name
    print "best classifier:", best_name,",accuracy result:", best_acc
    return best_classifier
    # print('BernoulliNB`s accuracy is %f' % score(BernoulliNB(), trainFeatures, testFeatures))
    #
    # print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB(), trainFeatures, testFeatures))
    #
    # print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression(), trainFeatures, testFeatures))
    #
    # print('SVC`s accuracy is %f' % score(SVC(), trainFeatures, testFeatures))
    #
    # print('LinearSVC`s accuracy is %f' % score(LinearSVC(), trainFeatures, testFeatures))
    #
    # print('NuSVC`s accuracy is %f' % score(NuSVC(), trainFeatures, testFeatures))


def split_data(posFeatures, negFeatures):
    shuffle(posFeatures)  # 把文本的排列随机化

    shuffle(negFeatures)  # 把文本的排列随机化

    print "pos len:", len(posFeatures), "; neg len:", len(negFeatures)

    # 3:1
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))

    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    print "train len:",len(trainFeatures), "; test len:", len(testFeatures)
    return trainFeatures, testFeatures


def group_by_date(data):
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)
    data = pd.DataFrame.sort_index(data)  # 从小到大排序
    unique_date =  data.index.unique()
    result = collections.OrderedDict()
    for d in unique_date:
        result[d] = list()
    for index, row in data.iterrows():
        result[index].append(row['comment'])
    return result

def predict(model, filename, informative_features):
    data = pd.read_csv(basePath + filename, sep='\t', header=None,
                                      names=['read_count', 'comment_count', 'comment', 'author', 'release_date'],index_col='release_date')
    grouped_data = group_by_date(data) #按日期分组的数据

    count_sentiment = collections.OrderedDict()
    for date in grouped_data.keys():
        count_sentiment[date] = dict({'pos':0,'neg':0})

    for date, comments in grouped_data.items():
        cut_comments = cut_words(comments)
        vector_comments = transform_data(cut_comments, informative_features)
        pred = model.classify_many(vector_comments)  # 对测试集的数据进行分类，给出预测的标签
        count_sentiment[date]['pos'] += pred.count('pos') #统计看涨数量
        count_sentiment[date]['neg'] += pred.count('neg') #统计看跌数量

    sentiment_result = pd.DataFrame()
    for date, sentiment in count_sentiment.items():
        # ssi = sentiment['pos'] - sentiment['neg'] # 简单情感指数
        bi = np.log(1.0*(1+sentiment['pos'])/(1+sentiment['neg'])) #看涨指数
        dis = np.abs(1 - np.abs(  1.0*(sentiment['pos']-sentiment['neg'])/(sentiment['pos']+sentiment['neg']) ) ) #情感差异指数

        row = collections.OrderedDict()
        row['date'] = date
        row['pos'] = sentiment['pos']
        row['neg'] = sentiment['neg']
        # row['ssi'] = ssi
        row['bi'] = bi+0.6 #统一加上
        row['dis'] = dis
        sentiment_result = sentiment_result.append(row,ignore_index=True)

    sentiment_result.to_csv(basePath+'szzs-sentiment-add0.6.csv',sep='\t',encoding='utf-8',index=False,float_format='%.2f',low_memory=False)

    # pd.DataFrame({'comment':original_data,'label':test_pred}).to_csv('../data/szzs-result.csv',sep='\t',encoding='utf-8',index=False)


if __name__ == '__main__':
    # informative_features = bag_of_words(text())#单个词

    # informative_features = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)#双个词

    # informative_features =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)#单个词和双个词

    informative_features = chi_features(800)  # 结巴分词

    posFeatures, negFeatures = build_features(informative_features)  # 获得训练数据

    trainFeatures, testFeatures = split_data(posFeatures, negFeatures)

    models = collections.OrderedDict({'BernoulliNB':BernoulliNB(),'MultinomialNB':MultinomialNB(),
                                      'LogisticRegression':LogisticRegression(),'SVC':SVC(),'LinearSVC':LinearSVC(),
                                      'NuSVC':NuSVC(),'Decision Tree':DecisionTreeClassifier(),'Random Forest':RandomForestClassifier()})
    best_classifier = compare_model(models,trainFeatures,testFeatures)

    predict(best_classifier, 'comment-szzs.csv', informative_features)
