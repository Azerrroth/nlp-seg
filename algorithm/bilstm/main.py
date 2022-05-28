import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import random
from data.dataset import make_dloader
from model.bilstm import BiLSTM

config = {
    "run_name": "PKU",
    "vocab_size": 1e7,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "lstm_dropout": 0.2,
    "label_size": 4,
    "batch_size": 64,
    "max_len": 128,
    "base_lr": 1e-3,
    "init_lr": 1e-10,
    "l2_coeff": 1e-6,
    "warmup_steps": 10,
    "decay_factor": 0.5,
    "seed": 42,
    "label2id": {
        'B': 0,
        'M': 1,
        'E': 2,
        'S': 3
    },
    "wandb": False,
    "early_stopping": True,
    "gradient_clip_norm": 0,
}


def create_callbacks(config):
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=
        f"./data/model_checkpoints/{config['run_name']}_{''.join([str(random.randint(0,9)) for _ in range(9)])}",
        monitor="train/loss",
        mode="min",
        filename=f"{config['run_name']}" + "{epoch:02d}-{train/loss:.2f}",
        save_top_k=1,
    )
    callbacks = [saving]

    if config['early_stopping']:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=5,
            ))

    if config['wandb']:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    return callbacks


def main(config):

    from pytorch_lightning import seed_everything
    seed_everything(config['seed'])

    if config['wandb']:
        import wandb
        project = os.getenv("WANDB_PROJECT")
        entity = os.getenv("WANDB_ACCT")
        log_dir = os.getenv("LOG_DIR")

        if log_dir is None:
            log_dir = "./data/logs"
            print(
                "Using default wandb log dir path: {}.\nThis can be adjusted with the environment variable `logs`"
                .format(log_dir))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        assert (
            project is not None and entity is not None
        ), "Please set environment variables `WANDB_ACCT` and `WANDB_PROJ` with \n\
                your wandb user/organization name and project title, respectively."

        experiment = wandb.init(
            project=project,
            entity=entity,
            config=config,
            dir=log_dir,
            reinit=True,
        )
        wan_config = wandb.config
        wandb.run.name = config['run_name']
        wandb.run.save()
        logger = pl.loggers.WandbLogger(experiment=experiment,
                                        save_dir="./data/logs")
        logger.log_hyperparams(wan_config)

    label2id = config["label2id"]
    train_dloader, train_dataset = make_dloader(
        'seg-data/training/pku_training.utf8',
        batch_size=config['batch_size'],
        label_encoder=label2id,
        max_size=1e7,
        sep=' ',
        shuffle=True)

    # test_data = None
    # test_dloader = make_dloader('seg-data/testing/pku_test.utf8',
    #                             batch_size=config['batch_size'],
    #                             label_encoder=label2id,
    #                             max_size=1e7,
    #                             sep=' ',
    #                             shuffle=True)

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
        word_decoder=train_dataset.word_decoder,
        base_lr=config['base_lr'],
        init_lr=config['init_lr'],
        l2_coeff=config['l2_coeff'],
        warmup_steps=config['warmup_steps'],
        decay_factor=config['decay_factor'],
    )

    callbacks = create_callbacks(config)

    # Train
    trainer = pl.Trainer(
        # precision=16,
        # amp_backend="native",
        auto_select_gpus=True,
        gpus=None,
        callbacks=callbacks,
        accelerator="dp",
        logger=logger if config["wandb"] else None,
        log_gpu_memory=True,
        gradient_clip_val=config["gradient_clip_norm"],
        gradient_clip_algorithm="norm",
        # overfit_batches=0,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        val_check_interval=1.0,
        max_epochs=100,
        fast_dev_run=10,
    )

    trainer.fit(model=bilstm, train_dataloader=train_dloader)


if __name__ == "__main__":
    main(config=config)