import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json

'''
参数：
vac_size-字和对应数字（id）的字典长度
x_data-每一句话对应的id，所组成的列表,这些列表的集合

函数作用：
1. 加载图结构及模型参数
2. 进行预测，返回预测结果(每个字SBEM的概率列表)
'''
def load_model_prediction(vac_size, x_data):
    tf.reset_default_graph()
    # 加载图结构及模型参数
    sess = tf.Session()
    saver = tf.train.import_meta_graph('/content/drive/MyDrive/pku_model_data/first_model.ckpt.meta')
    saver.restore(sess,'/content/drive/MyDrive/pku_model_data/first_model.ckpt')

    graph = tf.get_default_graph()
    y_pre = graph.get_tensor_by_name("a3:0")
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    # 预测
    result = []
    for everyX in x_data:
      temp = sess.run(y_pre, feed_dict={x: everyX, keep_prob: 0.5})
      temp = temp[0,:,:]
      result.append(temp)
    return result

# 函数作用：使用维特比算法得到最优路径，返回这句话对应的SBME
def viterbi(result, trans_pro):
    # 形成一个列表，这个列表的每一项是一个字典，每个字典代表一个字（key:SBME，value:这个字SBME的概率）
    nodes = [dict(zip(('S', 'B', 'M', 'E'), i)) for i in result]
    paths = nodes[0]

    for t in range(1, len(nodes)):
        path_old = paths.copy()
        paths = {}

        for i in nodes[t]:
            nows = {}
            for j in path_old:
                if j[-1] + i in trans_pro:
                    # nows[j + i] = path_old[j] + nodes[t][i] + trans_pro[j[-1] + i]
                    nows[j + i] = path_old[j] + nodes[t][i]
            pro, key = max([(nows[key], key) for key, value in nows.items()])
            paths[key] = pro
    best_pro, best_path = max([(paths[key], key) for key, value in paths.items()])
    return best_path

'''
参数：
txt-这句话
best_path-这句话对应的概率最大的SBME组合

函数作用：
根据这句话和它的SBME，分割出单词列表

输出：
分割出的单词列表
'''
def segword(txt, best_path):
    begin, end = 0, 0
    seg_word = []
    for index, char in enumerate(txt):
        signal = best_path[index]
        if signal == 'B':
            begin = index
        elif signal == 'E':
            seg_word.append(txt[begin:index + 1])
            end = index + 1
        elif signal == 'S':
            seg_word.append(char)
            end = index + 1
    if end < len(txt):
        seg_word.append(txt[end:])
    return seg_word

def cnn_seg(path):
    # 一个长度为4726的字典，key:某个字，value:这个字对应的数字（id）
    word_id = json.load(open('/content/drive/MyDrive/pku_vacabulary.json', 'r'))
    vacabulary_size = len(word_id) + 1

    # 前后两个字可能的标签组合
    trans_pro = {'SS': 1, 'BM': 1, 'BE': 1, 'SB': 1, 'MM': 1, 'ME': 1, 'EB': 1, 'ES': 1}


    # 读取测试文本
    fr = open(path, encoding='utf-8')
    # 文章中每一句话都转成id，所组成的列表
    artical2id = []
    # 文章中的每一句话组成的列表
    txts = []
    # 逐行操作
    for line in fr.readlines():
        # 获取每行文字
        txt = line.strip().split()[0]
        txts.append(txt)
        # 将这句话中的每个字转换成对应的数字（id）,如果没有这个字，默认转成4726
        txt2id = [[word_id.get(word, 4726) for word in txt]]
        # 将每行id加入到列表中
        artical2id.append(txt2id)
    fr.close()

    # 预测
    result = load_model_prediction(vacabulary_size, x_data=artical2id)

    # 将结果存储到文件中
    fw = open("/content/drive/MyDrive/result_data/pku_test_result.utf8", "w+", encoding='utf-8')
    for i in range(len(result)):
      # 获取每一句话的BSME列表
      best_path = viterbi(result[i], trans_pro)
      # 根据句子和它的BSME进行分词，获取分词的列表
      wordsList = segword(txts[i], best_path)
      # 存储
      for j in range(len(wordsList)):
        if j < len(wordsList)-1 :
          fw.write(wordsList[j] + "  ")
        else:
          fw.write(wordsList[j])
      fw.write("\n")
    fw.close()


if __name__ == '__main__':
    cnn_seg("/content/drive/MyDrive/corpus_data/pku_test.utf8")
