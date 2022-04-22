from cmath import inf
import math
import sys

import re
from tqdm import tqdm
from zhon.hanzi import punctuation

sys.path.append(".")
from analyzer import Analyzer
# from utils.chinese_utils import is_zh


class NGram(Analyzer):

    def __init__(self, n=2, smooth="Add-Delta", **kwargs):
        super().__init__()

        self.n = n
        self.smooth = smooth
        self.kwargs = kwargs
        self.total_freq = 0
        self.max_word_len = 5

        self.word_dict = {}  # Vocabulary
        self.bigram_dict = {}  # Bigram Word Frequency
        self.word_dict_freq = {}  # word occurrence probability
        self.bigram_trans_dict = {}

        self.start_token = "<BEG>"
        self.end_token = "<END>"

        self.word_dict[self.start_token] = 0
        self.word_dict[self.end_token] = 0

    def segment(self, sentence, **kwargs):
        '''
        Analyze the sentenec and return the segment result

        :param sentence: the sentence to be analyzed

        :return: the segment result
        '''
        sentence = sentence.strip()
        # 初始化
        node_state_list = []  # 主要是记录节点的最佳前驱，以及概率值总和
        ini_state = {}
        ini_state['pre_node'] = -1
        ini_state['prob_sum'] = -0  #当前概率总和
        node_state_list.append(ini_state)
        # 逐个节点的寻找最佳的前驱点
        for node in range(1, len(sentence) + 1):
            # 寻找最佳前驱，并记录当前最大的概率累加值
            (best_pre_node,
             best_prob_sum) = self.get_best_pre_nodes(sentence, node,
                                                      node_state_list)
            # 添加到队列
            cur_node = {"pre_node": best_pre_node, "prob_sum": best_prob_sum}
            node_state_list.append(cur_node)
        # 获得最优路径，从后到前
        best_path = []
        node = len(sentence)
        best_path.append(node)
        while True:
            pre_node = node_state_list[node]['pre_node']
            if pre_node == -1:
                break
            node = pre_node
            best_path.append(node)
        # 构建词的切分
        word_list = []
        for i in range(len(best_path) - 1, 0, -1):
            left = best_path[i]
            right = best_path[i - 1]

            word = sentence[left:right]
            word_list.append(word)

        return word_list

    def train(self, train_data: list, segment: str = " ", **kwargs):
        for sentence in tqdm(train_data, desc="Train"):
            self.to_word_dict(sentence=sentence.strip(), segment=segment)
            self.to_bigram_dict(sentence=sentence.strip(), segment=segment)
        # self.calc_word_freq()
        # self.calc_bigram_freq()

        # self.total_freq -= self.word_dict.pop(self.start_token)
        # self.total_freq -= self.word_dict.pop(self.end_token)

        # self.bigram_dict.pop(self.start_token)
        # self.bigram_dict.pop(self.end_token)

        return

    def to_word_dict(self, sentence: str, segment: str = " "):
        '''
        Convert the sentence to word dict

        :param sentence: the sentence to be converted

        :return: the word dict
        '''

        words = sentence.strip().split(segment)

        self.word_dict[self.start_token] += 1
        self.word_dict[self.end_token] += 1
        # self.total_freq += 2

        for word in words:
            word = word.strip()
            # Check if the word is Chinese
            # cleaned_data = ''.join(re.findall(r'[\u4e00-\u9fa5]', word))

            # if cleaned_data != '':
            if word not in self.word_dict.keys():
                self.word_dict[word] = 1
            else:
                self.word_dict[word] += 1
            self.total_freq += 1
            # self.max_word_len = max(self.max_word_len, len(word))
        return

    def to_bigram_dict(self, sentence: str, segment: str = ""):
        '''
        Convert the sentence to bigram dict

        :param sentence: the sentence to be converted

        :return: the bigram dict
        '''
        words = sentence.split(segment)
        for i, word in enumerate(words):
            word = word.strip()
            if i == 0:
                if self.start_token not in self.bigram_dict:
                    self.bigram_dict[self.start_token] = {}
                else:
                    if word not in self.bigram_dict[self.start_token]:
                        self.bigram_dict[self.start_token][word] = 1
                    else:
                        self.bigram_dict[self.start_token][word] += 1
            else:
                pre_word = words[i - 1]
                if pre_word not in self.bigram_dict:
                    self.bigram_dict[pre_word] = {word: 1}
                else:
                    if word not in self.bigram_dict[pre_word]:
                        self.bigram_dict[pre_word][word] = 1
                    else:
                        self.bigram_dict[pre_word][word] += 1

        # # End part
        # last_word = words[-1].strip()
        # if last_word in self.bigram_dict:
        #     if self.end_token not in self.bigram_dict[last_word]:
        #         self.bigram_dict[word][self.end_token] = 1
        #     else:
        #         self.bigram_dict[word][self.end_token] += 1
        # else:
        #     self.bigram_dict[last_word] = {self.end_token: 1}
        return

    def calc_word_freq(self):
        '''
        Calculate the word frequency

        :return: None
        '''
        for word in self.word_dict:
            self.word_dict_freq[word] = math.log(self.word_dict[word] /
                                                 self.total_freq)

    # Note: 平滑计算时，对于 pre_word, post_word 都未登录的情况，要使其概率最小，否则程序会趋近于使句子形成长词
    def smooth_probability(self, pre_word, post_word, verbose: bool = False):
        '''
        Smooth methods to deal with the unknown word

        :param pre_word: the previous word
        :param post_word: the next word

        :return: the smoothed probability
        '''
        if self.smooth == "Add-Delta" or self.smooth == "Add-One":
            if self.smooth == "Add-Delta":
                delta = self.kwargs["delta"]
            else:
                delta = 1

            if pre_word not in self.bigram_dict:
                print(
                    f"First not occur: {pre_word} {post_word}. {delta} / {self.total_freq} * {delta}"
                ) if verbose else None
                return delta / (self.total_freq * delta + self.total_freq)
            else:
                if post_word not in self.bigram_dict[pre_word]:
                    print(
                        f"First occur: {pre_word} {post_word}. {delta} / {len(self.bigram_dict[pre_word])} * {delta} + {self.word_dict[pre_word]}"
                    ) if verbose else None
                    return delta / (len(self.bigram_dict[pre_word]) * delta +
                                    self.word_dict[pre_word])
                else:
                    probablity = (self.bigram_dict[pre_word][post_word] + delta
                                  ) / (self.word_dict[pre_word] +
                                       len(self.bigram_dict[pre_word]) * delta)
                    print(
                        f"Both occur: {pre_word} {post_word}. Pro:{math.log(probablity)} {self.bigram_dict[pre_word][post_word]} + {delta} / {self.word_dict[pre_word]} + {len(self.bigram_dict[pre_word])} * {delta}"
                    ) if verbose else None
                    return probablity

    def get_word_trans_prob(self, pre_word, post_word):
        '''
        Get bigram trans probablity

        :param pre_word: the previous word
        :param post_word: the next word

        :return: the smoothed probability
        '''
        # trans_word = pre_word + " " + post_word
        if pre_word == self.start_token and post_word in self.word_dict:
            trans_prob = math.log(self.word_dict[post_word] / self.total_freq)
            return trans_prob * len(post_word)
        trans_prob = math.log(self.smooth_probability(
            pre_word, post_word)) * len(post_word)
        return trans_prob

    # 寻找node的最佳前驱节点，方法为寻找所有可能的前驱片段
    def get_best_pre_nodes(self, sentence, node, node_state_list):
        # 如果node比最大词小，则取的片段长度的长度为限
        max_seg_length = min([node, self.max_word_len])
        pre_node_list = []  # 前驱节点列表

        # 获得所有的前驱片段，并记录累加概率
        for segment_length in range(1, max_seg_length + 1):
            segment_start_node = node - segment_length
            segment = sentence[segment_start_node:node]  # 获取前驱片段
            pre_node = segment_start_node  # 记录对应的前驱节点
            if pre_node == 0:
                # 如果前驱片段开始节点是序列的开始节点，则概率为<S>转移到当前的概率
                pre_pre_word = self.start_token
                segment_prob = self.get_word_trans_prob(pre_pre_word, segment)
            else:  # 如果不是序列的开始节点，则按照二元概率计算
                # 获得前驱片段的一个词
                pre_pre_node = node_state_list[pre_node]["pre_node"]
                pre_pre_word = sentence[pre_pre_node:pre_node]
                if pre_pre_word == "":
                    pre_pre_word = self.start_token
                segment_prob = self.get_word_trans_prob(pre_pre_word, segment)
            pre_node_prob_sum = node_state_list[pre_node]["prob_sum"]
            # 当前 node 一个候选的累加概率值
            candidate_prob_sum = (pre_node_prob_sum) + (segment_prob)
            pre_node_list.append(
                (pre_node, candidate_prob_sum, pre_pre_word + " " + segment))

        # Find the max `candidate_prob_sum` in `pre_node_list`
        (best_pre_node, best_prob_sum, *_) = max(pre_node_list,
                                                 key=lambda d: d[1])
        return best_pre_node, best_prob_sum


def preprocess(sentence):
    sentence = sentence.strip()
    # sentence = sentence.replace(" ", "")
    # sentence = sentence.replace("\n", "")
    # sentence = sentence.replace("\t", "")
    # sentence = sentence.replace("\r", "")
    sentence = re.sub(r"[%s]+" % punctuation, "", sentence)  #去除， ！ ？ 。
    sentence = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]", "",
                      sentence)

    sentence = re.sub(r"\s\s+", "  ", sentence)

    return sentence


if __name__ == "__main__":

    dataset = "pku"  # (msra, pku)

    ngram = NGram(n=2, smooth="Add-Delta", delta=3e-20)
    train_data = open(f"seg-data/training/{dataset}_training.utf8",
                      "r",
                      encoding="utf-8")
    sentences = train_data.readlines()
    sent_list = []
    for sentence in sentences:
        sentence = preprocess(sentence)
        sent_list.append(sentence)
    ngram.train(train_data=sent_list, segment="  ")

    # Test Part
    test_data = open(f"seg-data/testing/{dataset}_test.utf8",
                     "r",
                     encoding="utf-8")
    test_sentences = test_data.readlines()
    with open(f"./output/{dataset}_test.utf8.seg", "w", encoding="utf-8") as f:
        for sentence in tqdm(test_sentences, desc="Testing"):
            sentence = preprocess(sentence)
            f.write("  ".join(ngram.segment(sentence)))
            f.write("\n")
    # for sentence in test_sentences:
    #     print(sentence.strip())
    #     sentence = preprocess(sentence)
    #     sentence = sentence.strip().replace("  ", "")
    #     print(sentence)
    #     print(ngram.segment(sentence))