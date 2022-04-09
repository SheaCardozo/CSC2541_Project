import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from jax.config import config
config.update("jax_enable_x64", True)

import haiku as hk
from haiku.nets import ResNet18
import jax
from jax import numpy as jnp

import optax

from sklearn.metrics import accuracy_score

import kfac_jax

from itertools import product

import argparse
import time
import json

from utils import get_datasets
from models import DCNet


def main (args) :

    dataset = args.dataset
    iterations = args.iterations
    model = args.model
    lr = args.lr
    folder = args.folder
    seed = args.seed
    batch_size = args.batch_size
    L2_REG = args.l2_reg

    train, test = get_datasets(dataset)

    gpus = jax.devices("gpu")
    cpus = jax.devices("cpu")

    key = jax.random.PRNGKey(seed=seed)
    key, key_samp, key_init = jax.random.split(key, 3)

    if model == "dcnet":
        classifier = hk.without_apply_rng(hk.transform_with_state(lambda x, training: DCNet()(x, training)))
    elif model == "resnet18":
        classifier = hk.without_apply_rng(hk.transform_with_state(lambda x, training: ResNet18(num_classes=10)(x, training)))
    else:
        raise ValueError("Not Implemented")

    if dataset == "mnist":
        sample = jax.random.normal(key_samp, shape=(9, 32, 32, 1))
    elif dataset == 'cifar10':
        sample = jax.random.normal(key_samp, shape=(9, 32, 32, 3))

    params_base, state_base = classifier.init(key_init, sample, training=True) 
    params_base, state_base = jax.device_put(params_base, device=cpus[0]), jax.device_put(state_base, device=cpus[0])

    test_img = test['image']
    test_y = test['label']

    def loss_fn(params, state, batch, labs):
        logits, state = classifier.apply(params, state, x=batch, training=True)
        return -jnp.mean(jnp.sum(jnp.log(jax.nn.softmax(logits)) * labs, axis=1)), state


    def get_grafted_train_step(optim_dir, optim_mag):
        @jax.jit
        def grafted_train_step(batch, labs, params, state, optim_state_dir, optim_state_mag, dir_kfac=None, mag_kfac=None, eps=1e-8):

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, state), grad = grad_fn(params, state, batch, labs)

            if dir_kfac is not None:
                updates_dir, optim_state_dir = dir_kfac
            elif optim_state_dir is not None:
                updates_dir, optim_state_dir = optim_dir.update(grad, optim_state_dir, params)

            if mag_kfac is not None:
                updates_mag, optim_state_mag = mag_kfac
                if optim_state_dir is None:
                    updates_dir = updates_mag
            else:
                updates_mag, optim_state_mag = optim_mag.update(grad, optim_state_mag, params)
            

            mag_update = jax.tree_multimap(lambda x, y: x / (y + eps), 
                                        jax.tree_util.tree_map(jnp.linalg.norm, updates_mag), 
                                        jax.tree_util.tree_map(jnp.linalg.norm, updates_dir))
            
            cos_sim = jax.tree_multimap(lambda x, y: (jnp.dot(x.reshape((-1)), y.reshape((-1))) / (jnp.linalg.norm(x.reshape((-1))) * jnp.linalg.norm(y.reshape((-1))))) , updates_mag, updates_dir)
        
            
            updates = jax.tree_multimap(lambda x, y: x * y, mag_update, updates_dir)


            params = optax.apply_updates(params, updates)

            return jnp.mean(loss), params, state, optim_state_dir, optim_state_mag, mag_update, cos_sim
        return grafted_train_step

    @jax.jit
    def predict(params, state, images):
        probs, _ = classifier.apply(params, state, x=images, training=False)
        preds = jnp.argmax(probs, axis=1)
        return preds

    def softmax_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray):
        kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
        return optax.softmax_cross_entropy(logits, targets)

    def loss_fn_wrapped(params, state, batch):
        batch, labs = batch
        logits, state = classifier.apply(params, state, x=batch, training=True)
        loss = jnp.mean(softmax_cross_entropy(logits, labs)) + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0

        return loss, state


    optims = [None, optax.adam(lr), optax.sgd(lr, momentum=0.9)]#, optax.adam(lr), None]
    optims_name = ['kfac', 'adam', 'sgd']# 'adam', 'kfac']

    log_loss = {}

    kfac_args = {
        "value_and_grad_func": jax.value_and_grad(loss_fn_wrapped, has_aux=True),
        "l2_reg": L2_REG,
        "value_func_has_aux": False,
        "value_func_has_state": True,
        "value_func_has_rng": False,
        "use_adaptive_learning_rate": True,
        "use_adaptive_momentum": False,
        "use_adaptive_damping": True,
        "initial_damping": 100.0,
        "min_damping": 10.0,
        "multi_device": False,
        "inverse_update_period": 1,
        "damping_adaptation_interval": 1
    }

    for (optim_dir, optim_dir_name), (optim_mag, optim_mag_name) in product(zip(optims, optims_name), zip(optims, optims_name)):
        
        start = time.time()

        grafted_train_step = get_grafted_train_step(optim_dir, optim_mag)

        def loop (optim_dir, optim_dir_name, optim_mag, optim_mag_name, log_loss, key):
            params, state = jax.device_put(params_base, device=gpus[0]), jax.device_put(state_base, device=gpus[0])

            log_loss[f'{optim_mag_name}#{optim_dir_name}'] = {}
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["losses"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["val_metric"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["rho"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["damping"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["lr"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["momentum"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["new_loss"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["loss"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["quad_model_change"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_ratio"] = []
            log_loss[f'{optim_mag_name}#{optim_dir_name}']["cos_sim"] = []

            if optim_mag_name == "kfac":
                optim_mag = kfac_jax.Optimizer(**kfac_args)
                if optim_dir_name == "kfac":
                    optim_dir = optim_dir
            elif optim_dir_name == "kfac":
                optim_dir = kfac_jax.Optimizer(**kfac_args)

            if optim_mag_name == "kfac":
                key, key_choice = jax.random.split(key)
                batch_inds = jax.random.choice(key=key_choice, a=train['image'].shape[0], shape=(batch_size,), replace=False).astype(jnp.int64)
                batch = jax.device_put(train['image'][batch_inds], device=gpus[0])
                labs = jax.nn.one_hot(jax.device_put(train['label'][batch_inds], device=gpus[0]).astype(jnp.int64), 10)     
                optim_state_mag = optim_mag.init(params, key, (batch, labs), func_state=state)
            else:
                optim_state_mag = optim_mag.init(params)

            if optim_dir_name == "kfac":
                if optim_mag_name == "kfac":
                    optim_state_dir = None
                else:
                    key, key_choice = jax.random.split(key)
                    batch_inds = jax.random.choice(key=key_choice, a=train['image'].shape[0], shape=(batch_size,), replace=False).astype(jnp.int64)
                    batch = jax.device_put(train['image'][batch_inds], device=gpus[0])
                    labs = jax.nn.one_hot(jax.device_put(train['label'][batch_inds], device=gpus[0]).astype(jnp.int64), 10)   
                    optim_state_dir = optim_dir.init(params, key, (batch, labs), func_state=state)
            else:
                optim_state_dir = optim_dir.init(params)

            for i in range(iterations):
                key, key_choice = jax.random.split(key)
                batch_inds = jnp.array(jax.random.choice(key=key_choice, a=train['image'].shape[0], shape=(batch_size,), replace=False)).astype(jnp.int64)
                batch = jax.device_put(train['image'][batch_inds], device=gpus[0])
                labs = jax.nn.one_hot(jax.device_put(train['label'][batch_inds], device=gpus[0]).astype(jnp.int64), 10)

                dir_kfac, mag_kfac = None, None

                if optim_mag_name == "kfac":
                    _, mag_update, state, optim_state_mag, mag_stats = optim_mag.step(params, optim_state_mag, key, batch=(batch, labs), global_step_int=i, func_state=state, momentum=0.9)
                    mag_kfac = mag_update, optim_state_mag
                    if optim_dir_name == "kfac":
                        dir_kfac = None
                elif optim_dir_name == "kfac":
                    _, dir_update, state, optim_state_dir, dir_stats = optim_dir.step(params, optim_state_dir, key, batch=(batch, labs), global_step_int=i, func_state=state, momentum=0.9)
                    dir_kfac = dir_update, optim_state_dir

                loss, params, state, optim_state_dir, optim_state_mag, mag_inflate, cos_sim = grafted_train_step(batch, labs, params, state, optim_state_dir, optim_state_mag, dir_kfac, mag_kfac)

                log_loss[f'{optim_mag_name}#{optim_dir_name}']["losses"].append(loss.item())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_ratio"].append(jnp.vstack(jax.tree_util.tree_flatten(mag_inflate)[0]).mean().item())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["cos_sim"].append(jnp.vstack(jax.tree_util.tree_flatten(cos_sim)[0]).mean().item())

                if optim_mag_name == "kfac":
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["rho"].append(mag_stats['rho'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["damping"].append(mag_stats['damping'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["lr"].append(mag_stats['learning_rate'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["momentum"].append(mag_stats['momentum'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["new_loss"].append(mag_stats['new_loss'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["loss"].append(mag_stats['loss'].item())
                    log_loss[f'{optim_mag_name}#{optim_dir_name}']["quad_model_change"].append(mag_stats['quad_model_change'].item())

                test_inds = jnp.array(jax.random.choice(key=key_choice, a=test_img.shape[0], shape=(batch_size,), replace=False)).astype(jnp.int64)
                
                preds = predict(params, state, jax.device_put(test_img[test_inds], device=gpus[0]))

                log_loss[f'{optim_mag_name}#{optim_dir_name}']["val_metric"].append(accuracy_score(jax.device_put(test_y[test_inds], device=gpus[0]), preds))

            return log_loss, key

        log_loss, key = loop (optim_dir, optim_dir_name, optim_mag, optim_mag_name, log_loss, key)

        end = time.time()

        print(f"{optim_mag_name}#{optim_dir_name}, Time: {end - start}")

        
        with open(f'/w/284/sacardoz/run/{folder}/log.txt', 'w') as f:
            f.write(json.dumps(log_loss)) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_reg', type=float, default=1e-3)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--seed', type=int, default=451)
    parser.add_argument('--batch_size', type=int, default=2048)

    args = parser.parse_args()
    main (args)
