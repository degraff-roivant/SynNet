import time
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.nn import functional as F

from syn_net.models.v2 import utils


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 4,
        hidden_dim: int = 1000,
        num_layers: int = 5,
        dropout: float = 0.5,
        num_dropout_layers: int = 1,
        task: str = "classification",
        loss: str = "cross_entropy",
        valid_loss: str = "accuracy",
        out_feat: Optional[str] = None,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        val_freq: int = 10,
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        for i in range(num_layers - 2):
            block = [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            layers.extend(block)
            if i > num_layers - 3 - num_dropout_layers:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if task == "classification":
            layers.append(nn.Softmax())

        self.ffn = nn.Sequential(*layers)
        self.criterion = utils.get_loss(loss)
        self.metric = utils.get_metric(valid_loss)
        self.kdtree = utils.build_tree(out_feat)

        self.optimizer = optimizer
        self.lr = learning_rate
        self.val_freq = val_freq

    def forward(self, x):
        return self.ffn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.ffn(x)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, True, True, False, True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.current_epoch % self.val_freq == 0:
            x, y = batch
            y_hat = self.ffn(x)

            val_loss = self.metric(y_hat, y, self.kdtree)
            self.log("val_loss", val_loss, True, True, False, True)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), self.lr)

        return optimizer


def load_array(data_arrays, batch_size, is_train=True, ncpu=-1):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=is_train, num_workers=ncpu
    )


if __name__ == "__main__":
    states_list = []
    steps_list = []
    for i in range(1):
        states_list.append(
            np.load(
                "/home/rociomer/data/synth_net/pis_fp/states_" + str(i) + "_valid.npz",
                allow_pickle=True,
            )
        )
        steps_list.append(
            np.load(
                "/home/rociomer/data/synth_net/pis_fp/steps_" + str(i) + "_valid.npz",
                allow_pickle=True,
            )
        )

    states = np.concatenate(states_list, axis=0)
    steps = np.concatenate(steps_list, axis=0)

    X = states
    y = steps[:, 0]

    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(y)

    batch_size = 64
    train_data_iter = load_array((X_train, y_train), batch_size, is_train=True)

    pl.seed_everything(0)
    mlp = MLP()
    tb_logger = TensorBoardLogger("temp_logs/")

    trainer = pl.Trainer(
        gpus=[0], max_epochs=30, progress_bar_refresh_rate=20, logger=tb_logger
    )
    t = time.time()
    trainer.fit(mlp, train_data_iter, train_data_iter)
    print(time.time() - t, "s")
