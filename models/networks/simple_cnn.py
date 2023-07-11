import flax.linen as nn


class SimpleEncoder(nn.Module):
    c_hid: int
    latent_dim: int
    act_fn: str
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        act_fn = getattr(nn.activation, self.act_fn)
        while x.shape[1] > 4:
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 32x32 => 16x16
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)
        return x


class SimpleDecoder(nn.Module):
    c_hid: int
    latent_dim: int
    act_fn: str
    batch_norm: bool = False
    max_img_size: int = 32

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        act_fn = getattr(nn.activation, self.act_fn)
        x = nn.Dense(features=4 * 4 * self.c_hid)(x)
        x = x.reshape(x.shape[0], 4, 4, self.c_hid)  # Single feature vector to image grid
        while x.shape[1] < self.max_img_size:
            x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(
                x
            )  # 4x4 => 8x8
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = act_fn(x)
        x = nn.Conv(features=3, kernel_size=(3, 3))(x)
        x = nn.tanh(x)
        return x
