from analyzer import Analyzer
'''
平滑策略：

- 加法平滑: 

'''


class NGram(Analyzer):
    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def analyze(self, sentence, **kwargs):
        '''
        Analyze the sentenec and return the segment result

        :param sentence: the sentence to be analyzed

        :return: the segment result
        '''

        return []

    def train(self, train_data, **kwargs):

        return
