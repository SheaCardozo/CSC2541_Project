import haiku as hk
import jax
from jax import numpy as jnp

class DCNet(hk.Module):
    def __init__(self):
        super().__init__(name="DCNet")
        self.conv1 = hk.Conv2D(output_channels=64*1, kernel_shape=4, stride=2, padding="SAME", with_bias=False)
        self.bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, data_format="N...C")
        self.conv2 = hk.Conv2D(output_channels=64*2, kernel_shape=4, stride=2, padding="SAME", with_bias=False)
        self.bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, data_format="N...C")
        self.conv3 = hk.Conv2D(output_channels=64*4, kernel_shape=4, stride=2, padding="SAME", with_bias=False)
        self.bn3 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, data_format="N...C")
        self.conv4 = hk.Conv2D(output_channels=64*8, kernel_shape=4, stride=2, padding="SAME", with_bias=False)
        self.bn4 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, data_format="N...C")

        self.conv5 = hk.Conv2D(output_channels=64*16, kernel_shape=2, stride=1, padding="valid")
        self.fc = hk.Linear(10, with_bias=True)

    def __call__(self, x, training):
        x = self.conv1(x)
        x = self.bn1(x, is_training=training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = self.bn2(x, is_training=training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = self.bn3(x, is_training=training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)

        x = self.conv4(x)
        x = self.bn4(x, is_training=training)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)


        x = self.conv5(x)
        x = jnp.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x