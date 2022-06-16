from datetime import date
from email.mime import base
import os
import sys

from tqdm import tqdm
import yaml

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
from model.bert import Bert
from utils.tag2seg import tag2seg
from utils.cut_list import cut_list

base_config = {
    "run_name": "PKU_0dropout",
    "model_name": "bilstm",  # (bert, bilstm)
    "vocab_size": 1e7,
    "token_length": 1000,
    "embedding_dim": 10,
    "hidden_dim": 256,
    "num_layers": 3,
    "lstm_dropout": 0,
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
    "wandb": True,
    "wandb_project": "nlp-bilstm",
    "wandb_entity": "azerrroth",
    "early_stopping": True,
    "gradient_clip_norm": 0,
    "save_dir": "./output",
    "num_workers": 16,
    "train_data_path": "seg-data/training/pku_training.utf8",
    "test_data_path": "seg-data/testing/pku_test.utf8",
    "gold_data_path": "seg-data/gold/pku_test_gold.utf8",
}


def create_models(config, label_decoder):
    if config["model_name"] == "bert":
        model = Bert(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            lstm_dropout=config["lstm_dropout"],
            label_size=config["label_size"],
            batch_size=config["batch_size"],
            max_len=config["max_len"],
            label_decoder=label_decoder,
            bert_model_name=config["bert_model_name"],
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            base_lr=config["base_lr"],
            init_lr=config["init_lr"],
            l2_coeff=config["l2_coeff"],
            warmup_steps=config["warmup_steps"],
            decay_factor=config["decay_factor"],
        )
    elif config["model_name"] == "bilstm":
        model = BiLSTM(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            lstm_dropout=config["lstm_dropout"],
            label_size=config["label_size"],
            batch_size=config["batch_size"],
            max_len=config["max_len"],
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            label_decoder=label_decoder,
            base_lr=config["base_lr"],
            init_lr=config["init_lr"],
            l2_coeff=config["l2_coeff"],
            warmup_steps=config["warmup_steps"],
            decay_factor=config["decay_factor"],
        )
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")

    return model


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


def predict(model, sentence: str, token_length: int, dataset):
    words_arr = sentence.strip()
    words_arr = list(words_arr)
    sentence_vec = torch.LongTensor(1, token_length).fill_(0)
    mask = torch.ByteTensor(1, token_length).fill_(0)
    for i, word in enumerate(words_arr):
        sentence_vec[0, i] = dataset.decode_word(word)
    mask[0, :len(sentence)] = torch.tensor([1] * len(sentence),
                                           dtype=torch.bool)

    tag_preds = model.predict(sentence_vec, mask)
    # tag_preds = model.crf.decode(tag_preds, mask)
    tag_preds = np.array(tag_preds).flatten()
    tags = [dataset.decode_label(x) for x in tag_preds]
    return tag2seg(sentence, tags)


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
        limit_max_len=config['limit_max_len'],
        num_workers=config["num_workers"],
        model_name=config["model_name"],
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
        limit_max_len=config['limit_max_len'],
        num_workers=config["num_workers"],
        model_name=config["model_name"],
    )

    model = create_models(config, train_dataset.label_decoder)

    callbacks = create_callbacks(config, model_save_dir)

    # Train
    trainer = pl.Trainer(
        precision=32,
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
        accumulate_grad_batches=10,
        sync_batchnorm=True,
        val_check_interval=1.0,
        max_epochs=100,
        default_root_dir=f"{config['save_dir']}/checkpoints",
    )

    trainer.fit(model=model,
                train_dataloaders=train_dloader,
                val_dataloaders=test_dloader)

    # Test
    # Get best model
    model_checkpoint_callback = callbacks[0]
    best_model_path = model_checkpoint_callback.best_model_path

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # best_model = BiLSTM.load_from_checkpoint(checkpoint_path=best_model_path, )
    trainer.test(model=model, dataloaders=test_dloader)

    output_path = "output/{}.{}.txt".format(config['run_name'], date.today())
    output_file = open(output_path, "w")
    with open(config["test_data_path"], "r") as test_data:
        for sentence in tqdm(test_data.readlines(), desc="Predicting"):
            if len(sentence) > config['token_length'] and config.get(
                    "limit_max_len", False):

                subsentence = cut_list(list(sentence), config['token_length'])
                res = ""
                for item in subsentence:
                    res += predict(model,
                                   "".join(item),
                                   config['token_length'],
                                   dataset=train_dataset)
            else:
                res = predict(model,
                              sentence,
                              config['token_length'],
                              dataset=train_dataset)
            output_file.write(res.strip() + "\n")
    output_file.close()

    case = "今天天气怎么样？"
    print(predict(model, case, config['token_length'], dataset=train_dataset))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default="algorithm/bilstm/config/bert_pku.yml",
                        help="path to config file")

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        for key in base_config:
            if config.get(key, None) is None:
                config[key] = base_config[key]
                print("{} is not in config file, use default value {}".format(
                    key, base_config[key]))
    else:
        config = base_config

    print(config)
    main(config=config)

    pass