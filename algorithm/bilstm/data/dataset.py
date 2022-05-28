import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Dataset

from process import get_vocabulary, process_data


# 建立词表，将词表中的词转换为数字索引
# 文本标签表 标记为 B E M S
class SegDataset(Dataset.Dataset):
    def __init__(self,
                 datapath: str,
                 label_encoder: dict,
                 max_size=1e7,
                 sep: str = ' '):
        # self.super().__init__()

        self.datapath = datapath
        word_list, label_list = process_data(datapath, sep=sep)
        word_encoder = get_vocabulary(word_list=word_list, max_size=max_size)
        self.word_encoder = word_encoder
        self.word_decoder = {v: k for k, v in self.word_encoder.items()}
        self.label_encoder = label_encoder
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        self.dataset = self.preprocess(word_list, label_list)

    def preprocess(self, words_list, labels_list):
        """convert the data to ids"""
        processed = []
        for (words, labels) in zip(words_list, labels_list):
            word_id = [self.word_encoder[word] for word in words]  # word to id
            label_id = [self.label_encoder[label]
                        for label in labels]  # label to id
            processed.append((word_id, label_id))
        print("-------- Process Done! --------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, words, labels, batch_size):
        token_len = max([len(x) for x in labels])
        word_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(words, labels)):
            word_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]),
                                                      dtype=torch.uint8)

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
        return self.word_encoder.get(word)

    def decode_word(self, word_id):
        return self.word_decoder.get(word_id)


def make_dloader(datapath,
                 batch_size,
                 label_encoder,
                 max_size=1e7,
                 sep=' ',
                 shuffle: bool = True):
    dataset = SegDataset(datapath, label_encoder, max_size, sep)
    dloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          collate_fn=dataset.collate_fn)
    return dloader, dataset


if __name__ == '__main__':
    label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    file_path = 'seg-data/training/pku_training.utf8'
    loader, train_dataset = make_dloader(file_path,
                                         batch_size=32,
                                         label_encoder=label2id,
                                         max_size=1e8,
                                         sep=' ')