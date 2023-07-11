from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import random
from jax_trainer.datasets import Batch
from jax_trainer.trainer import TrainerModule
from jax_trainer.trainer.trainer import TrainState


class AutoencoderTrainer(TrainerModule):
    def batch_to_input(self, batch: Batch) -> Any:
        return batch.input

    def loss_function(
        self, params: Any, state: TrainState, batch: Batch, rng: random.PRNGKey, train: bool = True
    ) -> Tuple[Any, Tuple[Any, Dict]]:
        x = self.batch_to_input(batch)
        x_hat, mutable_variables = self.model_apply(params, state, x, rng, train=train)
        loss = jnp.mean((x - x_hat) ** 2)
        return loss, (mutable_variables, {})
