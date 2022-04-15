from abc import ABC, abstractmethod

import numpy as np


class Analyzer(ABC):

    def __init__(self):
        # super.__init__()
        pass

    @abstractmethod
    def analyze(self, sentence, **kwargs):
        '''
        Analyze the sentenec and return the segment result

        :param sentence: the sentence to be analyzed

        :return: the segment result
        '''

        return []

    @abstractmethod
    def train(self, train_data, train_dict, **kwargs):
        '''
        Train the model

        :param train_data: 预先划分的句子
        :
        :param **kwargs:

        :return: None
        '''

        return