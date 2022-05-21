import os
import logging
import numpy as np


def getlist(input_str):
    '''
    将输入词转换为 B E M S 标记

    B: begin
    E: end
    M: middle
    S: single
    '''
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'E']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('E')
    return output_str


def process_data(file_path: str,
                 max_size: int = None,
                 output_path: str = None,
                 sep: str = '  '):
    '''
    处理数据
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError('{} not found'.format(file_path))
    word_list = []
    label_list = []
    num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            num += 1
            words = []
            line = line.strip()
            if not line:
                continue  # line is None
            for i in range(len(line)):
                if line[i] == " ":
                    continue  # skip space
                words.append(line[i])
            # print(words)
            word_list.append(words)
            text = line.split(sep)
            # print(text)
            labels = []
            for item in text:
                if item == "":
                    continue
                labels.extend(getlist(item))
            # print(labels)
            label_list.append(labels)
            assert len(labels) == len(words), "labels 数量与 words 不匹配"
        print("We have", num, "lines in", file_path, "file processed")
        # 保存成二进制文件
        # np.savez_compressed(output_dir, words=word_list, labels=label_list)
        logging.info(
            "-------- {} data process DONE!--------".format(file_path))

    return word_list, label_list


def get_vocabulary(
    word_list: list,
    max_size: int = 1e7,
):
    '''
    获取词汇表
    '''

    # 如果有处理好的，就直接load
    # 如果没有处理好的二进制文件，就处理原始的npz文件
    word_freq = {}
    for line in word_list:
        for ch in line:
            if ch in word_freq:
                word_freq[ch] += 1
            else:
                word_freq[ch] = 1
    index = 0
    sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)
    word_encoder = {}
    # 构建 word2id 字典
    for elem in sorted_word:
        word_encoder[elem[0]] = index
        index += 1
        if index >= max_size:
            break
    # id2word 保存
    # word_decoder = {_idx: _word for _word, _idx in list(word_encoder.items())}
    # 保存为二进制文件
    # np.savez_compressed(self.vocab_path,
    #                     word2id=word_encoder,
    #                     id2word=self.id2word)
    print("-------- Vocabulary Build! --------")

    return word_encoder  # , word_decoder


if __name__ == '__main__':
    file_path = 'seg-data/training/pku_training.utf8'
    word_list, label_list = process_data(file_path, sep=' ')
    print(len(word_list))
    print(len(label_list))
    word_encoder, word_decoder = get_vocabulary(file_path, word_list)