import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Dataset

from process import get_vocabulary, process_data
from transformers import BertTokenizer


# 建立词表，将词表中的词转换为数字索引
# 文本标签表 标记为 B E M S
class SegDataset(Dataset.Dataset):
    def __init__(
        self,
        datapath: str,
        label_encoder: dict,
        max_size=1e7,
        sep: str = ' ',
        train: bool = True,
        word_encoder: dict = None,
        token_length: int = 1000,
        limit_max_len: bool = False,
        model_name: str = "bert",
    ):
        # self.super().__init__()

        self.datapath = datapath
        self.token_length = token_length
        self.limit_max_len = limit_max_len
        self.model_name = model_name
        if model_name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",
                                                           do_lower_case=True)
        word_list, label_list = process_data(datapath, sep=sep)
        if word_encoder is None:
            word_encoder = get_vocabulary(word_list=word_list,
                                          max_size=max_size)
        self.word_encoder = word_encoder
        self.word_decoder = {v: k for k, v in self.word_encoder.items()}
        self.label_encoder = label_encoder
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        self.dataset = self.preprocess(word_list, label_list)

    def check_exists(self, word):
        """
        动态增加词表，因为之前没有，词频肯定是最低的，直接加到最后
        """
        exists = self.word_encoder.get(word)
        if exists is None:
            print("Add word to vocabulary: {}. Vocab size: {}".format(
                word, len(self.word_encoder)))
            self.word_encoder[word] = len(self.word_encoder)
            self.word_decoder[len(self.word_decoder)] = word

    def preprocess(self, words_list, labels_list):
        """convert the data to ids"""
        processed = []
        for (words, labels) in zip(words_list, labels_list):
            for word in words:
                self.check_exists(word)

            word_id = [self.encode_word(word) for word in words]  # word to id
            label_id = [self.encode_label(label)
                        for label in labels]  # label to id

            if self.limit_max_len and len(word_id) > self.token_length:
                new_word_id = self.cut_list(word_id, self.token_length)
                new_label_id = self.cut_list(label_id, self.token_length)
                splited_processed = list(zip(new_word_id, new_label_id))
                processed.extend(splited_processed)
            else:
                processed.append((word_id, label_id))
        print("Finish preprocess")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, words, labels, batch_size):
        # assert self.token_length >= max([len(x) for x in labels])
        if self.limit_max_len:
            token_len = self.token_length
        else:
            token_len = max(self.token_length, max([len(x) for x in labels]))

        word_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.BoolTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(words, labels)):
            word_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]),
                                                      dtype=torch.bool)

        return word_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):

        words = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in labels]
        batch_size = len(batch)

        word_ids, label_ids, input_mask = self.get_long_tensor(
            words, labels, batch_size)

        return word_ids, label_ids, input_mask, lens

    def encode_label(self, label):
        return self.label_encoder.get(label)

    def decode_label(self, label_id):
        return self.label_decoder.get(label_id)

    def encode_word(self, word):
        if self.model_name == "bert":
            return self.tokenizer.convert_tokens_to_ids(word)
        return self.word_encoder.get(word)

    def decode_word(self, word_id):
        if self.model_name == "bert":
            return self.tokenizer.convert_ids_to_tokens(word_id)
        return self.word_decoder.get(word_id)

    def cut_list(self, lists, cut_len):
        """
        将列表拆分为指定长度的多个列表
        :param lists: 初始列表
        :param cut_len: 每个列表的长度
        :return: 一个二维数组 [[x,x],[x,x]]
        """
        res_data = []
        if len(lists) > cut_len:
            for i in range(int(len(lists) / cut_len)):
                cut_a = lists[cut_len * i:cut_len * (i + 1)]
                res_data.append(cut_a)

            last_data = lists[int(len(lists) / cut_len) * cut_len:]
            if last_data:
                res_data.append(last_data)
        else:
            res_data.append(lists)

        return res_data


def make_dloader(datapath,
                 batch_size,
                 label_encoder,
                 model_name: str,
                 max_size=1e7,
                 sep=' ',
                 train: bool = True,
                 word_encoder=None,
                 token_length: int = 1000,
                 limit_max_len: bool = True,
                 num_workers: int = 32,
                 shuffle: bool = False):
    dataset = SegDataset(
        datapath=datapath,
        label_encoder=label_encoder,
        max_size=max_size,
        sep=sep,
        train=train,
        word_encoder=word_encoder,
        token_length=token_length,
        limit_max_len=limit_max_len,
        model_name=model_name,
    )
    dloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
    )
    return dloader, dataset


if __name__ == '__main__':
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    file_path = 'seg-data/training/pku_training.utf8'
    loader, train_dataset = make_dloader(
        file_path,
        batch_size=32,
        label_encoder=label2id,
        max_size=1e7,
        sep=' ',
        model_name='bert',
    )
