import pickle
import tensorflow_datasets as tfds
import jax
from jax import numpy as jnp

def unpickle(file):
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d

def get_datasets(dataset):
    cpus = jax.devices("cpu")

    if dataset == "mnist":
        ds_builder = tfds.builder('mnist')
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
        train_ds['image'] = (((jax.device_put(jnp.array(train_ds['image']), device=cpus[0]) / 255) - 0.5) / 0.5)
        test_ds['image'] = (((jax.device_put(jnp.array(test_ds['image']), device=cpus[0]) / 255) - 0.5) / 0.5)

        train_ds['image'] = jax.image.resize(train_ds['image'], (train_ds['image'].shape[0], 32, 32, 1), method='cubic').astype(jnp.float64)
        test_ds['image'] = jax.image.resize(test_ds['image'], (test_ds['image'].shape[0], 32, 32, 1), method='cubic').astype(jnp.float64)

        return train_ds, test_ds
    elif dataset == 'cifar10':
        X = None
        y = None
        for i in range(1, 6):
            df = unpickle(f"/w/284/sacardoz/cifar-10-batches-py/data_batch_{i}")

            if X is None:
                X = jax.device_put(jnp.array(df[b'data']), device=cpus[0])
            else:
                X = jnp.vstack([X, jax.device_put(jnp.array(df[b'data']), device=cpus[0])])
            
            if y is None:
                y = df[b'labels']
            else:
                y.extend(df[b'labels'])

        X = (((X.reshape((-1, 32, 32, 3))/ 255) - 0.5) / 0.5)

        train = {"image": X, "label": jax.device_put(jnp.array(y), device=cpus[0])}
        df = unpickle(f"/w/284/sacardoz/cifar-10-batches-py/test_batch")
        test_X = jax.device_put(jnp.array(df[b'data'].reshape((-1, 32, 32, 3))), device=cpus[0])
        test_X = (((test_X.reshape((-1, 32, 32, 3)) / 255) - 0.5) / 0.5)
        test_y = jax.device_put(jnp.array(df[b'labels']), device=cpus[0])
        test = {"image": test_X, "label": test_y}
        return train, test
    else:
        raise ValueError("Not Implemented")

