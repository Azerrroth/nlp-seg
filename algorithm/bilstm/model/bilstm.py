import os
import sys

print(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchcrf import CRF
import numpy as np
import utils.metrics as metrics


class BiLSTM(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        lstm_dropout: float,
        label_size: int,
        batch_size: int,
        max_len: int,
        label_decoder: dict,
        word_decoder: dict,
        device: torch.device = torch.device('cpu'),
        base_lr: float = 1e-3,
        init_lr: float = 1e-10,
        l2_coeff: float = 1e-6,
        warmup_steps: int = 10,
        decay_factor: float = 0.5,
    ):
        super(BiLSTM, self).__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout
        self.label_size = label_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.model_device = device
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.l2_coeff = l2_coeff
        self.warmup_steps = warmup_steps
        self.decay_factor = 0.5
        self.label_decoder = label_decoder
        self.word_decoder = word_decoder
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

    def forward(self, word, label, mask):
        embedding_res = self.embedding(word)
        lstm_out, _ = self.lstm(embedding_res)
        tag_preds = self.tag_pred(lstm_out)
        return tag_preds

    # def step(self, batch, train: bool = True):

    def configure_optimizers(self):
        weight_decay = 1e-6  # l2正则化系数
        # 同样，如果只有一个网络结构，就可以更直接了
        optimizer = optim.Adam(self.parameters(),
                               lr=self.base_lr,
                               weight_decay=self.l2_coeff)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            # init_lr=self.init_lr,
            # peak_lr=self.base_lr,
            # warmup_steps=self.warmup_steps,
            patience=2,
            factor=self.decay_factor,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': "train/loss"
        }

    def training_step(self, batch, batch_idx):
        word, label, mask, length = batch
        tag_preds = self(word, label, mask)
        crf_loss = self.crf(tag_preds, label, mask) * -1
        loss = crf_loss

        labels_pred = self.crf.decode(tag_preds, mask=mask)

        targets = [
            itag[:ilen] for itag, ilen in zip(label.cpu().numpy(), length)
        ]

        return {
            'loss': loss,
            'crf_loss': crf_loss,
            'labels_pred': labels_pred,
            'targets': targets
        }

    def training_step_end(self, output):
        self._log_stats('train', output)
        return output

    def training_epoch_end(self, outputs):
        true_tags = []
        pred_tags = []
        loss = 0
        for output in outputs:
            loss += output['loss'].item()
            targets = output['targets']
            labels_pred = output['labels_pred']
            true_tags.extend([[self.label_decoder.get(idx) for idx in indexs]
                              for indexs in targets])
            pred_tags.extend([[self.label_decoder.get(idx) for idx in indexs]
                              for indexs in labels_pred])

        f1, pre, rec = metrics.f1_score(true_tags, pred_tags)

        res = {
            "f1": f1,
            "precision": pre,
            "recall": rec,
            "loss": loss / len(outputs)
        }
        self._log_stats('train', res)

    def validation_step(self, batch, batch_idx):
        word, label, mask, length = batch
        tag_preds = self(word, label, mask)
        crf_loss = self.crf(tag_preds, label, mask) * -1
        loss = crf_loss
        labels_pred = self.crf.decode(tag_preds, mask=mask)
        targets = [
            itag[:ilen] for itag, ilen in zip(label.cpu().numpy(), length)
        ]
        return {
            'loss': loss,
            'crf_loss': crf_loss,
            'labels_pred': labels_pred,
            'targets': targets
        }

    def validation_step_end(self, output):
        self._log_stats('valid', output)
        return output

    def validation_epoch_end(self, outputs):
        true_tags = []
        pred_tags = []
        loss = 0
        for output in outputs:
            loss += output['loss'].item()
            targets = output['targets']
            labels_pred = output['labels_pred']
            true_tags.extend([[self.label_decoder.get(idx) for idx in indexs]
                              for indexs in targets])
            pred_tags.extend([[self.label_decoder.get(idx) for idx in indexs]
                              for indexs in labels_pred])

        f1, pre, rec = metrics.f1_score(true_tags, pred_tags)

        res = {
            "f1": f1,
            "precision": pre,
            "recall": rec,
            "loss": loss / len(outputs)
        }
        self._log_stats('valid', res)

    def _log_stats(self, section, outs):
        for key in outs.keys():
            if "loss" not in key:
                continue
            stat = outs[key]
            if isinstance(stat, np.ndarray) or isinstance(stat, torch.Tensor):
                stat = stat.mean()
            self.log(f"{section}/{key}", stat, sync_dist=False)

    def predict(self, word, mask):
        tag_preds = self(word, None, mask)
        labels_pred = self.crf.decode(tag_preds, mask=mask)
        return labels_pred