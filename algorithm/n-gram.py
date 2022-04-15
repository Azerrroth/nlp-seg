import sys

import nltk
import re

sys.path.append(".")
from analyzer import Analyzer
# from utils.chinese_utils import is_zh
'''
平滑策略：

- 加法平滑: 


//代码框架
void _find (int cur)//找划分，cur表示当前为第几个词
{
	if (cur==n+1)	calc();//对当前结果计算概率并保存
	for (int i=cur;i<=n;++i)
		if (check(cur,i)){//如果从cur到i形成一个词
			add(cur,i);//将该词添加到当前划分
			_find(i+1);
			del();//删掉这个词
		}
}

'''


class NGram(Analyzer):

    def __init__(self, n=2):
        super().__init__()
        self.n = n
        self.word_dict = {}

    def analyze(self, sentence, **kwargs):
        '''
        Analyze the sentenec and return the segment result

        :param sentence: the sentence to be analyzed

        :return: the segment result
        '''

        return []

    def train(self, train_data, **kwargs):

        return

    def to_word_dict(self, sentence: str, segment: str = " "):
        '''
        Convert the sentence to word dict

        :param sentence: the sentence to be converted

        :return: the word dict
        '''

        words = sentence.split(segment)
        for word in words:
            cleaned_data = ''.join(re.findall(r'[\u4e00-\u9fa5]', word))

            if cleaned_data != '':
                if word not in self.word_dict:
                    self.word_dict[word] = 1
                else:
                    self.word_dict[word] += 1
        return

    def to_bigram_dict(self, sentence: str, segment: str = ""):
        '''
        Convert the sentence to bigram dict

        :param sentence: the sentence to be converted

        :return: the bigram dict
        '''
        start_token = "<BEG>"
        end_token = "<END>"

        words = sentence.split(segment)
        for i, word in enumerate(words):

            if i == 0:
                if start_token not in self.word_dict:
                    self.word_dict[start_token] = {}
                else:
                    self.word_dict[start_token][
                        word] = 1 if word not in self.word_dict[
                            start_token] else self.word_dict[start_token][
                                word] + 1
            else:
                last_word = words[i - 1]
                if word not in self.word_dict:
                    self.word_dict[word] = {last_word: 1}
                else:
                    self.word_dict[word][
                        last_word] = 1 if last_word not in self.word_dict[
                            word] else self.word_dict[word][last_word] + 1
        # End part
        last_word = words[-1]
        if last_word in self.word_dict:
            if end_token not in self.word_dict[last_word]:
                self.word_dict[word][end_token] = 1
            else:
                self.word_dict[word][end_token] += 1
        else:
            self.word_dict[last_word] = {end_token: 1}
        return


if __name__ == "__main__":
    ngram = NGram()
    train_data = open("seg-data/training/msr_training.utf8",
                      "r",
                      encoding="utf-8")
    for line in train_data.readlines():
        ngram.to_bigram_dict(line, "  ")
    print(ngram.word_dict)