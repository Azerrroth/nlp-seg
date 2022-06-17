import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import re
import numpy as np
import json
from collections import Counter, defaultdict
import tqdm
import os


# 输入单词，输出对应的sbme标签
def makelabel(word):
    if len(word) == 1:
        return 'S'
    else:
        return 'B' + (len(word) - 2) * 'M' + 'E'


# 预处理训练样本，处理结果为：按从长到短的顺序排列的单词列表和sbme列表
def get_corpus(path):
    labels = []
    texts = []

    stops = u'，。！？；、：“,"\'\.!\?;:\n'
    with open(path, 'r', encoding='utf-8') as f:
        # 按符号和换行符切割所有非空行，所得内容结果里面只包含空格和单词
        txt = [line.strip(' ') for line in re.split('[' + stops + ']', f.read()) if line.strip(' ')]

        # 将每一句中的单词和它们对应的标签存到列表中（每一句为一项）
        for line in txt:
            curWord = ''
            curLable = ''
            for word in re.split(' +', line):  # 按一个或多个空格分割
                curWord += word
                curLable += makelabel(word)

            texts.append(curWord)
            labels.append(curLable)

    # 从长到短的下标顺序
    ls = [len(i) for i in texts]
    ls = np.argsort(ls)[::-1]  # 从大到小排序。argsort()表示对数据进行从小到大进行排序，返回数据的索引值；[::-1]倒着排列

    # 将两个列表都按从长到短排列
    texts = [texts[i] for i in ls]
    labels = [labels[i] for i in ls]
    return texts, labels


# 做一个生成器，用来生成每个batch的训练样本。
# 这里的batch_size只是一个上限，
# 因为要求每个batch内的句子长度都要相同，这样子并非每个batch的size都能达到batch_size
def data(texts, labels, word_id, tag2vec, batch_size=256):
    l = len(texts[0])
    x = []
    y = []
    for i in range(len(texts)):
        # 如果句子长度发生改变 或者 本batch中句子的数量已达上限，则新开一个batch
        if len(texts[i]) != l or len(x) == batch_size:
            yield x, y
            x = []
            y = []
            l = len(texts[i])
        x.append([word_id[j] for j in texts[i]])
        y.append([tag2vec[j] for j in labels[i]])


def cnn_train(pure_txts, pure_tags, word_id, tag2vec, epoch=300):
    tf.reset_default_graph()

    embedding_size = 128
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    embeddings = tf.Variable(tf.random_uniform([vacabulary_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

    x = tf.placeholder(tf.int32, shape=[None, None], name="x")
    embedded = tf.nn.embedding_lookup(embeddings, x)
    embedded_dropout = tf.nn.dropout(embedded, keep_prob)

    W1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size], -1.0, 1.0), dtype=tf.float32, name="W1")
    b1 = tf.Variable(tf.random_uniform([embedding_size], -1.0, 1.0), dtype=tf.float32, name="b1")
    a1 = tf.nn.relu(tf.nn.conv1d(embedded_dropout, W1, stride=1, padding='SAME') + b1, name="a1")

    W2 = tf.Variable(tf.random_uniform([3, embedding_size, int(embedding_size / 4)], -1.0, 1.0), name="W2")
    b2 = tf.Variable(tf.random_uniform([int(embedding_size / 4)], -1.0, 1.0), name="b2")
    a2 = tf.nn.relu(tf.nn.conv1d(a1, W2, stride=1, padding='SAME') + b2, name="a2")

    W3 = tf.Variable(tf.random_uniform([3, int(embedding_size / 4), 4], -1.0, 1.0), name="W3")
    b3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0), name="b3")
    a3 = tf.nn.softmax(tf.nn.conv1d(a2, W3, stride=1, padding='SAME') + b3, name="a3")

    # 用交叉熵作为损失函数
    y_ = tf.placeholder(tf.float32, shape=[None, None, 4], name="y_")
    cross_entropy = -tf.reduce_sum(y_ * tf.log(a3 + 1e-20))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # 两行配合，计算准确率
    # 利用tf.argmax()按行求出真实值y_、预测值a3最大值的下标，用tf.equal()求出真实值和预测值相等的数量，也就是预测结果正确的数量
    correct_prediction = tf.equal(tf.argmax(a3, 2), tf.argmax(y_, 2))
    # tf.cast 强制类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(epoch):
        temp_data = tqdm.tqdm(data(pure_txts, pure_tags, word_id, tag2vec, batch_size=512),
                              desc=u'Epoch %s,Accuracy:0.0' % (i + 1))
        k = 0
        accs = []
        for x_data, y_data in temp_data:
            k += 1
            if k % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: x_data, y_: y_data, keep_prob: 1})
                accs.append(acc)
                temp_data.set_description('Epoch %s, Accuracy: %s' % (i + 1, acc))
            sess.run(train_step, feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
        print(u'Epoch %s Mean Accuracy: %s' % (i + 1, np.mean(accs)))

    saver = tf.train.Saver()
    saver.save(sess, '/content/drive/MyDrive/pku_model_data/first_model.ckpt')


# 统计每个字出现的次数，按从多到少排列(输出出现次数大于等于2的字的相关信息)。
# word_count字及个数；word_id字及编号1-n；vacabulary_size字的种类
def word2dic(pure_txts, flat=True):
    min_count = 2
    word_count = Counter(''.join(pure_txts))
    word_count = Counter({word: index for word, index in word_count.items() if index >= min_count})
    word_id = defaultdict(int)
    id = 0
    for i in word_count.most_common():
        id += 1
        word_id[i[0]] = id
    vacabulary_size = len(word_id) + 1
    if flat:
        json.dump(word_id, open('/content/drive/MyDrive/pku_vacabulary.json', 'w'))
    return word_count, word_id, vacabulary_size


if __name__ == '__main__':
    texts, labels = get_corpus('/content/drive/MyDrive/corpus_data/pku_training.utf8')

    word_count, word_id, vacabulary_size = word2dic(texts, flat=True)

    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}

    cnn_train(texts, labels, word_id, tag2vec, epoch=200)