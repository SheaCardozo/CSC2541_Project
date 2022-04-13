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
        ds_builder = tfds.image_classification.Cifar10()
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(
            ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(
            ds_builder.as_dataset(split='test', batch_size=-1))
        
        train_ds['image'] = (((jax.device_put(jnp.array(train_ds['image']), device=cpus[0]) / 255) - 0.5) / 0.5)
        test_ds['image'] = (((jax.device_put(jnp.array(test_ds['image']), device=cpus[0]) / 255) - 0.5) / 0.5)

        train_ds['image'] = train_ds['image'].astype(jnp.float64)
        test_ds['image'] = test_ds['image'].astype(jnp.float64)

        return train_ds, test_ds
    else:
        raise ValueError("Not Implemented")

def setup_log (log_loss, optim_dir_name, optim_mag_name):

    log_loss[f'{optim_mag_name}#{optim_dir_name}'] = {}
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["losses"] = []
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["val_metric"] = []
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_m"] = []
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_d"] = []
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_inflate"] = []
    log_loss[f'{optim_mag_name}#{optim_dir_name}']["cos_sim"] = []

    return log_loss