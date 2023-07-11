from typing import Any, Optional

import jax
import numpy as np
from jax import random
from jax_trainer.callbacks import Callback
from jax_trainer.datasets import DatasetModule
from ml_collections import ConfigDict


class ReconstructionCallback(Callback):
    def __init__(
        self, config: ConfigDict, trainer: Any, data_module: Optional[DatasetModule] = None
    ):
        super().__init__(config, trainer, data_module)
        self.n_samples = self.config.get("n_samples", 4)
        self.input_batch = trainer.batch_to_input(trainer.exmp_input)[: self.n_samples]
        self.create_reconstruction_function()

    def create_reconstruction_function(self):
        @jax.jit
        def reconstruct(state, x):
            x_hat, _ = self.trainer.model_apply(
                state.params, state, x, random.PRNGKey(0), train=False
            )
            return x_hat

        self.reconstruct = reconstruct

    def on_filtered_training_epoch_end(self, train_metrics, epoch_idx):
        x_rec = self.reconstruct(self.trainer.state, self.input_batch)
        x_rec = jax.device_get(x_rec)
        x_comb = np.stack([self.input_batch, x_rec], axis=2)
        x_comb = np.pad(x_comb, ((0, 0), (2, 2), (0, 0), (2, 2), (0, 0)), constant_values=0)
        x_comb = np.reshape(
            x_comb,
            (x_comb.shape[0], x_comb.shape[1], x_comb.shape[2] * x_comb.shape[3], x_comb.shape[4]),
        )
        x_comb = (x_comb + 1.0) / 2.0
        for i in range(x_comb.shape[0]):
            self.trainer.logger.log_image(f"reconstruction_{i}", x_comb[i], epoch_idx)
