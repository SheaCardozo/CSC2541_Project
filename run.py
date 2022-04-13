import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_log_compiles", True)

#import warnings
#warnings.filterwarnings("ignore")

import haiku as hk
from haiku.nets import ResNet18

import jax
from jax import numpy as jnp

import argparse
import json
import kfac_jax
import optax
import os
import pickle
import time

from sklearn.metrics import accuracy_score


from utils import get_datasets, setup_log
from models import DCNet
from hf.optimizer import hf

def main (args) :

    model = args.model
    dataset = args.dataset

    iterations = args.iterations
    lr = args.lr
    batch_size = args.batch_size

    checkpoints = args.checkpoints
    layer_wise = args.layer_wise

    seed = args.seed

    folder = args.folder
    L2_REG = args.l2_reg

    init_damp_kfac = 10.0
    min_damp_kfac = 10.0

    init_damp_hf = 10.0
    min_damp_hf = 10.0

    gpus = jax.devices("gpu")
    cpus = jax.devices("cpu")

    train, test = get_datasets(dataset)

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
    else:
        raise ValueError("Not Implemented")

    params_base, state_base = classifier.init(key_init, sample, training=True) 
    params_base, state_base = jax.device_put(params_base, device=cpus[0]), jax.device_put(state_base, device=cpus[0])

    test_img = test['image']
    test_y = test['label']

    # Some important functions for the training loop
    @jax.jit
    def predict(params, state, images):
        probs, _ = classifier.apply(params, state, x=images, training=False)
        preds = jnp.argmax(probs, axis=1)
        return preds

    def softmax_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray):
        kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
        return optax.softmax_cross_entropy(logits, targets)

    def kfac_loss_fn(params, state, batch):
        batch, labs = batch
        logits, state = classifier.apply(params, state, x=batch, training=True)
        loss = jnp.mean(softmax_cross_entropy(logits, labs)) + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0

        return loss, state

    @jax.jit
    def loss_fn(params, state, batch, labs):
        logits, state = classifier.apply(params, state, x=batch, training=True)
        return -jnp.mean(jnp.sum(jnp.log(jax.nn.softmax(logits)) * labs, axis=1)), state

    jit_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def get_grafted_train_step(optim_dir, optim_mag):
        @jax.jit
        def grafted_train_step(batch, labs, params, state, optim_state_dir, optim_state_mag, dir_kfac=None, mag_kfac=None, eps=1e-8):

            if dir_kfac is None or mag_kfac is None:
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, state), grad = grad_fn(params, state, batch, labs)
            else:
                loss, state = loss_fn(params, state, batch, labs)

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
            

            if layer_wise:
                mag_update = jax.tree_multimap(lambda x, y: x / (y + eps), 
                                            jax.tree_util.tree_map(jnp.linalg.norm, updates_mag), 
                                            jax.tree_util.tree_map(jnp.linalg.norm, updates_dir))
            else:
                global_update = jnp.linalg.norm(jnp.vstack(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: x.reshape((-1, 1)), updates_mag))[0])) / \
                                                    (jnp.linalg.norm(jnp.vstack(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: x.reshape((-1, 1)), updates_dir))[0])) + eps)

                mag_update = jax.tree_util.tree_map(lambda x: global_update, updates_mag)
            
            
            cos_sim = jax.tree_multimap(lambda x, y: (jnp.dot(x.reshape((-1)), y.reshape((-1))) / \
                        (eps + jnp.linalg.norm(x.reshape((-1))) * jnp.linalg.norm(y.reshape((-1))))), \
                        updates_mag, updates_dir)
        
            
            updates = jax.tree_multimap(lambda x, y: x * y, mag_update, updates_dir)


            params = optax.apply_updates(params, updates)

            return jnp.mean(loss), params, state, optim_state_dir, optim_state_mag, \
                jax.tree_util.tree_map(jnp.linalg.norm, updates_mag), \
                jax.tree_util.tree_map(jnp.linalg.norm, updates_dir), cos_sim

        return grafted_train_step
        
    kfac_args = {
        "value_and_grad_func": jax.value_and_grad(kfac_loss_fn, has_aux=True),
        "l2_reg": L2_REG,
        "value_func_has_aux": False,
        "value_func_has_state": True,
        "value_func_has_rng": False,
        "use_adaptive_learning_rate": True,
        "use_adaptive_momentum": True,
        "use_adaptive_damping": True,
        "initial_damping": init_damp_kfac,
        "min_damping": min_damp_kfac,
        "multi_device": False,
        "inverse_update_period": 1,
        "damping_adaptation_interval": 1
    }

    hf_args = {
        "precond": "uncentered",
        "lambd": init_damp_hf,
        "min_damp": min_damp_hf,
        "use_momentum": False,
        "line_search": True
    }
    
    mags_optims_name = ['sgd', 'adam']
    mags_optims = [optax.sgd(lr, momentum=0.9), optax.adam(lr)]
    dirs_optims_name = ['kfac', 'kfac']
    dirs_optims = [None, None]

    for (optim_dir, optim_dir_name), (optim_mag, optim_mag_name) in zip(zip(dirs_optims, dirs_optims_name), zip(mags_optims, mags_optims_name)):
        os.mkdir(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}")
        log_loss = {}

        start = time.time()

        grafted_train_step = get_grafted_train_step(optim_dir, optim_mag)

        # Segment the train loop in it's own function - this was aimed to force memory 
        # used in one run to be released for the next, but it matters less in Colab
        def loop (optim_dir, optim_dir_name, optim_mag, optim_mag_name, log_loss, key):
            params, state = jax.device_put(params_base, device=gpus[0]), jax.device_put(state_base, device=gpus[0])

            log_loss = setup_log(log_loss, optim_dir_name, optim_mag_name)

            # If kfac and hf are treated differently as they do not
            # conform to Optax api, setup them separately
            if optim_mag_name == "kfac":
                optim_mag = kfac_jax.Optimizer(**kfac_args)
                if optim_dir_name == "kfac":
                    optim_dir = None
            elif optim_dir_name == "kfac":
                optim_dir = kfac_jax.Optimizer(**kfac_args)

            if optim_mag_name == "hf":
                optim_mag = hf(classifier, loss_fn)
                if optim_dir_name == "hf":
                    optim_dir = None
            elif optim_dir_name == "hf":
                optim_dir = hf(classifier, loss_fn)

            if optim_mag_name == "kfac":
                key, key_choice = jax.random.split(key)
                batch_inds = jax.random.choice(key=key_choice, a=train['image'].shape[0], shape=(batch_size,), replace=False).astype(jnp.int64)
                batch = jax.device_put(train['image'][batch_inds], device=gpus[0])
                labs = jax.nn.one_hot(jax.device_put(train['label'][batch_inds], device=gpus[0]).astype(jnp.int64), 10)  
                optim_state_mag = optim_mag.init(params, key, (batch, labs), func_state=state)
            elif optim_mag_name == "hf":
                optim_state_mag = optim_mag.init(params, **hf_args)
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
            elif optim_dir_name == "hf":
                if optim_mag_name == "hf":
                    optim_state_dir = None
                else:
                    optim_state_dir = optim_dir.init(params, **hf_args)
            else:
                optim_state_dir = optim_dir.init(params)
            
            # Train loop
            for i in range(iterations + 1):

                # Batch
                key, key_choice = jax.random.split(key)
                batch_inds = jnp.array(jax.random.choice(key=key_choice, a=train['image'].shape[0], shape=(batch_size,), replace=False)).astype(jnp.int64)
                batch = jax.device_put(train['image'][batch_inds], device=gpus[0])
                labs = jax.nn.one_hot(jax.device_put(train['label'][batch_inds], device=gpus[0]).astype(jnp.int64), 10)

                # As with before - treat kfac and hf separately. Compute their 
                # update steps and pass them manually into the train step
                dir_kfac, mag_kfac = None, None

                if optim_mag_name == "kfac":
                    _, mag_update, state, optim_state_mag, mag_stats = \
                    optim_mag.step(params, optim_state_mag, key, batch=(batch, labs), global_step_int=i, func_state=state)
                    mag_kfac = mag_update, optim_state_mag
                    if optim_dir_name == "kfac":
                        dir_kfac = mag_update, optim_state_mag
                elif optim_dir_name == "kfac":
                    _, dir_update, state, optim_state_dir, dir_stats = \
                    optim_dir.step(params, optim_state_dir, key, batch=(batch, labs), global_step_int=i, func_state=state)
                    dir_kfac = dir_update, optim_state_dir


                if optim_mag_name == "hf" or optim_dir_name == "hf":

                    (_, state), hf_grad = jit_grad_fn(params, state, batch, labs)

                    if optim_mag_name == "hf":

                        mag_update, optim_state_mag = optim_mag.update(hf_grad, optim_state_mag, params, state, batch, labs)
                        mag_kfac = mag_update, optim_state_mag
                        if optim_dir_name == "hf":
                            dir_kfac = mag_update, optim_state_mag

                    elif optim_dir_name == "hf":
                        dir_update, optim_state_dir = optim_dir.update(hf_grad, optim_state_dir, params, state, batch, labs)
                        dir_kfac = dir_update, optim_state_dir

                # Train step
                loss, params, state, optim_state_dir, optim_state_mag, mag_m, mag_d, cos_sim = \
                    grafted_train_step(batch, labs, params, state, optim_state_dir, optim_state_mag, dir_kfac, mag_kfac)

                # Log iteration metrics
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["losses"].append(loss.item())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_m"].append(jnp.ravel(jnp.vstack(jax.tree_util.tree_flatten(mag_m)[0])).tolist())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_d"].append(jnp.ravel(jnp.vstack(jax.tree_util.tree_flatten(mag_d)[0])).tolist())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["mag_inflate"].append(jnp.ravel(jnp.vstack(jax.tree_util.tree_flatten(mag_m)[0]) / (1e-8 + jnp.vstack(jax.tree_util.tree_flatten(mag_d)[0]))).tolist())
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["cos_sim"].append(jnp.ravel(jnp.vstack(jax.tree_util.tree_flatten(cos_sim)[0])).tolist())

                # Compute and log validation metric (prediction accuracy)
                test_inds = jnp.array(jax.random.choice(key=key_choice, a=test_img.shape[0], shape=(batch_size,), replace=False)).astype(jnp.int64)
                preds = predict(params, state, jax.device_put(test_img[test_inds], device=gpus[0]))
                log_loss[f'{optim_mag_name}#{optim_dir_name}']["val_metric"].append(accuracy_score(jax.device_put(test_y[test_inds], device=gpus[0]), preds))

                # Save checkpoint if necessary
                if i % checkpoints == 0:
                    with open(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}/params_{optim_mag_name}#{optim_dir_name}_{i}.pkl", 'wb') as f:
                        pickle.dump(params, f)

                    with open(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}/state_{optim_mag_name}#{optim_dir_name}_{i}.pkl", 'wb') as f:
                        pickle.dump(state, f)

                    with open(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}/optim_state_dir_{optim_mag_name}#{optim_dir_name}_{i}.pkl", 'wb') as f:
                        pickle.dump(optim_state_dir, f)

                    with open(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}/optim_state_mag_{optim_mag_name}#{optim_dir_name}_{i}.pkl", 'wb') as f:
                        pickle.dump(optim_state_mag, f)
                
                    with open(f"/w/284/sacardoz/run/{folder}/{optim_mag_name}#{optim_dir_name}/log_loss_{optim_mag_name}#{optim_dir_name}_{i}.txt", 'w') as f:
                        f.write(json.dumps(log_loss)) 

                    print(f"{optim_mag_name}#{optim_dir_name}, Checkpoint: {i}")

            return log_loss, key

        # Run train loop
        log_loss, key = loop (optim_dir, optim_dir_name, optim_mag, optim_mag_name, log_loss, key)

        end = time.time()

        print(f"{optim_mag_name}#{optim_dir_name}, Time: {end - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_reg', type=float, default=0)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--seed', type=int, default=451)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--checkpoints', type=int, default=10)
    parser.add_argument('--layer-wise', type=bool, default=True)

    args = parser.parse_args()
    main (args)
