import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcrf import CRF


class BiLSTM(pl.LightningModule):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lstm_dropout: float,
                 label_size: int,
                 batch_size: int,
                 max_len: int,
                 device: torch.device = torch.device('cpu')):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout
        self.label_size = label_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=self.lstm_dropout,
                            batch_first=True,
                            bidirectional=True)

        self.tag_pred = nn.Linear(self.hidden_dim * 2, self.label_size)
        # 输出模型对 标记输出的概率 （未归一化）

        self.crf = CRF(self.label_size, batch_first=True)

    def forward(
        self,
        x,
    ):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.tag_pred(x)
        x = self.crf(x, mask)
        return x