from datetime import date
import os
from statistics import mode
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import random
import numpy as np
from data.dataset import make_dloader
from model.bilstm import BiLSTM
from utils.tag2seg import tag2seg

config = {
    "run_name": "MSR",
    "vocab_size": 1e7,
    "token_length": 1000,
    "embedding_dim": 10,
    "hidden_dim": 256,
    "num_layers": 3,
    "lstm_dropout": 0.2,
    "label_size": 4,
    "label2id": {
        'B': 0,
        'M': 1,
        'E': 2,
        'S': 3
    },
    "batch_size": 256,
    "max_len": 128,
    "base_lr": 1e-2,
    "init_lr": 1e-10,
    "l2_coeff": 1e-6,
    "warmup_steps": 10,
    "decay_factor": 0.5,
    "seed": 42,
    "wandb": False,
    "wandb_project": "nlp-bilstm",
    "wandb_entity": "azerrroth",
    "early_stopping": True,
    "gradient_clip_norm": 0,
    "save_dir": "./output",
    "num_workers": 16,
    "train_data_path": "seg-data/training/msr_training.utf8",
    "test_data_path": "seg-data/testing/msr_test.utf8",
    "gold_data_path": "seg-data/gold/msr_test_gold.utf8",
}


def create_callbacks(config, model_save_dir):
    saving = ModelCheckpoint(
        dirpath=f"{config['save_dir']}/model_checkpoints/{model_save_dir}",
        monitor="valid_f1",
        mode="max",
        save_last=True,
        filename=f"{config['run_name']}" +
        "{epoch:02d}-{valid_loss:.2f}-{valid_f1:.2f}",
        save_top_k=3,
    )
    callbacks = [saving]

    if config['early_stopping']:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="valid_f1",
                mode="max",
                patience=10,
            ))

    if config['wandb']:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    return callbacks


def predict(model, sentence: str, vocab: dict, token_length: int,
            label_decoder: dict):
    words_arr = sentence.strip()
    words_arr = list(words_arr)
    sentence_vec = torch.LongTensor(1, token_length).fill_(0)
    mask = torch.ByteTensor(1, token_length).fill_(0)
    for i, word in enumerate(words_arr):
        sentence_vec[0, i] = vocab.get(word, 0)
    mask[0, :len(sentence)] = torch.tensor([1] * len(sentence),
                                           dtype=torch.bool)

    tag_preds = model.predict(sentence_vec, mask)
    # tag_preds = model.crf.decode(tag_preds, mask)
    tag_preds = np.array(tag_preds).flatten()
    tags = [label_decoder.get(x) for x in tag_preds]
    return tag2seg(sentence, tags).strip()


def main(config):

    from pytorch_lightning import seed_everything
    model_save_dir = f"{config['run_name']}_{time.strftime('%m-%dT%H:%M', time.localtime())}"
    seed_everything(config['seed'])
    if config['wandb']:
        project = config.get('wandb_project')
        entity = config.get('wandb_entity')
        # project = os.getenv("WANDB_PROJECT")
        # entity = os.getenv("WANDB_ACCT")
        log_dir = os.getenv("LOG_DIR")

        if log_dir is None:
            log_dir = f"{config['save_dir']}/logs"
            print(
                "Using default wandb log dir path: {}.\nThis can be adjusted with the environment variable `logs`"
                .format(log_dir))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        assert (
            project is not None and entity is not None
        ), "Please set environment variables `WANDB_ACCT` and `WANDB_PROJ` with \n\
                your wandb user/organization name and project title, respectively."

        logger = WandbLogger(project=project,
                             name=config['run_name'],
                             config=config,
                             dir=log_dir,
                             reinit=True,
                             log_model=False,
                             save_dir=f"{config['save_dir']}/logs")
        logger.log_hyperparams(logger.experiment.config)

    label2id = config["label2id"]
    train_dloader, train_dataset = make_dloader(
        datapath=config['train_data_path'],
        batch_size=config['batch_size'],
        label_encoder=label2id,
        max_size=1e7,
        sep=' ',
        shuffle=True,
        train=True,
        token_length=config['token_length'],
        num_workers=config["num_workers"],
    )

    test_dloader, test_dataset = make_dloader(
        datapath=config['gold_data_path'],
        batch_size=config['batch_size'],
        label_encoder=label2id,
        max_size=1e7,
        sep=' ',
        shuffle=False,
        train=False,
        word_encoder=train_dataset.word_encoder,
        token_length=config['token_length'],
        num_workers=config["num_workers"],
    )

    bilstm = BiLSTM(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        lstm_dropout=config['lstm_dropout'],
        label_size=config['label_size'],
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        label_decoder=train_dataset.label_decoder,
        base_lr=config['base_lr'],
        init_lr=config['init_lr'],
        l2_coeff=config['l2_coeff'],
        warmup_steps=config['warmup_steps'],
        decay_factor=config['decay_factor'],
    )
    # print(pl.utilities.model_summary.summarize(bilstm, max_depth=2))

    callbacks = create_callbacks(config, model_save_dir)

    # Train
    trainer = pl.Trainer(
        precision=16,
        amp_backend="native",
        auto_select_gpus=True,
        gpus=1,
        callbacks=callbacks,
        strategy='dp',
        logger=logger if config["wandb"] else None,
        log_gpu_memory=True,
        gradient_clip_val=config["gradient_clip_norm"],
        gradient_clip_algorithm="norm",
        # overfit_batches=0,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        val_check_interval=1.0,
        max_epochs=100,
        default_root_dir=f"{config['save_dir']}/checkpoints",
    )

    trainer.fit(model=bilstm,
                train_dataloaders=train_dloader,
                val_dataloaders=test_dloader)

    # Test
    # Get best model
    model_checkpoint_callback = callbacks[0]
    best_model_path = model_checkpoint_callback.best_model_path

    checkpoint = torch.load(best_model_path)
    bilstm.load_state_dict(checkpoint["state_dict"])

    # best_model = BiLSTM.load_from_checkpoint(checkpoint_path=best_model_path, )
    trainer.test(model=bilstm, dataloaders=test_dloader)

    output_path = "output/{}.{}.txt".format(config['run_name'], date.today())
    output_file = open(output_path, "w")
    with open(config["test_data_path"], "r") as test_data:
        for sentence in tqdm(test_data.readlines(), desc="Predicting"):
            res = predict(bilstm, sentence, train_dataset.word_encoder,
                          config['token_length'], train_dataset.label_decoder)
            output_file.write(res + "\n")
    output_file.close()

    case = "今天天气怎么样？"
    print(
        predict(bilstm, case, train_dataset.word_encoder,
                config['token_length'], train_dataset.label_decoder))


if __name__ == "__main__":
    main(config=config)