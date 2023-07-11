import flax.linen as nn
from jax_trainer.utils import resolve_import_from_string
from ml_collections import FrozenConfigDict


class Autoencoder(nn.Module):
    encoder_config: FrozenConfigDict
    decoder_config: FrozenConfigDict

    def setup(self):
        encoder_class = resolve_import_from_string(self.encoder_config.name)
        self.encoder = encoder_class(**self.encoder_config.hparams)
        decoder_class = resolve_import_from_string(self.decoder_config.name)
        self.decoder = decoder_class(**self.decoder_config.hparams)

    def __call__(self, x, train=True, **kwargs):
        z = self.encoder(x, train=train, **kwargs)
        x_hat = self.decoder(z, train=train, **kwargs)
        return x_hat

    def encode(self, x, train=True, **kwargs):
        z = self.encoder(x, train=train, **kwargs)
        return z
