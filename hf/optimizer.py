from .utils import *
from collections import namedtuple
from jax import jit, grad, jvp, vjp, value_and_grad
from functools import partial

import optax
import jax
import jax.numpy as np


def hf(clf, loss, dloss=None):
    if dloss is None:  # If a derivative function is not supplied, take it here.
        @partial(jit, static_argnames=('is_training',))
        def dloss(params, state, batch, labels, is_training=True):
            return value_and_grad(loss, has_aux=True)(
                params, state, batch, labels, is_training)

    @jit
    def network_function(params, state, batch):
        return clf.apply(params, state, None, batch, True)[0]

    @jit
    def loss_logits(logits, labels):
        return np.mean(optax.softmax_cross_entropy(logits, labels))

    @jit
    def loss_hvp(logits, labels, v):
        def loss_z(z):
            return loss_logits(z, labels)

        return jvp(grad(loss_z), (logits,), (v,))[1]

    @jit
    def gnhvp(params, state, batch, labels, v):
        def f_net(w):
            return network_function(w, state, batch)

        z, R_z = jvp(f_net, (params,), (v,))
        R_gz = loss_hvp(z, labels, R_z)
        _, f_vjp = vjp(f_net, params)
        return f_vjp(R_gz)[0]

    @jit
    def dampened(params, state, batch, labels, v, lambd):
        R_G = gnhvp(params, state, batch, labels, v)
        return lin_comb(R_G, lambd, v)

    @jit
    def accumulate_sum_of_grad_squared(params, state, batch, labels):
        def body_fun(i, val):
            g = dloss(
                params, state, batch[i].reshape(-1, *batch[i].shape),
                labels[i].reshape(-1, *labels[i].shape))[1]
            return lin_comb(val, 1 / len(batch), hadamard(g, g))

        return jax.lax.fori_loop(0, len(batch), body_fun, zero_vec(params))

    @jit
    def Minv_factory_centered(params, state, batch, labels, lambd, alpha):
        diag = accumulate_sum_of_grad_squared(params, state, batch, labels)
        diag = lin_comb(diag, lambd, one_vec(params))
        return tree_map(lambda x: np.power(x, -1 * alpha), diag)

    @jit
    def Minv_factory_uncentered(batch_grad, lambd, alpha):
        diag = hadamard(batch_grad, batch_grad)
        diag = tree_map(lambda x: x+lambd, diag)
        # diag = lin_comb(
        #     hadamard(batch_grad, batch_grad), lambd, one_vec(batch_grad))
        return tree_map(lambda x: np.power(x, -1 * alpha), diag)

    @jit
    def cg_update_loop(x, r, z, p, params, state, batch, labels, Minv, lambd):
        alpha = dot(r, z) / dot(p, dampened(
            params, state, batch, labels, p, lambd))
        x = lin_comb(x, alpha, p)
        r_p = lin_comb(r, -alpha, dampened(
            params, state, batch, labels, p, lambd))
        z_p = hadamard(Minv, r_p)
        beta = dot(r_p, z_p) / dot(r, z)

        r = r_p  # r_{k+1} := r_k
        z = z_p  # z_{k+1} := z_k

        p = lin_comb(z, beta, p)

        return x, r, z, p

    def cg(
            lambd, b, x0, Minv, params, state, batch, labels,
            max_iter=50, epsilon=5e-4, fname='cg.txt'):
        x = x0
        r = lin_comb(b, -1, dampened(params, state, batch, labels, x, lambd))
        z = hadamard(Minv, r)
        p = z

        it = 0

        # Record information for stopping criteria
        phis = [0.5 * dot(x, dampened(
            params, state, batch, labels, x, lambd)) - dot(b, x)]

        # Record information for CG iteration backtracking
        chosen_ind = 0
        saved_params = [params]
        corr_losses = [loss(params, state, batch, labels)[0]]

        while True:
            x, r, z, p = cg_update_loop(
                x, r, z, p, params, state, batch, labels, Minv, lambd)
            phis.append(0.5 * dot(x, dampened(
                params, state, batch, labels, x, lambd)) - dot(b, x))
            it += 1

            saved_params.append(x)
            corr_losses.append(loss(
                lin_comb(params, 1.0, x), state, batch, labels)[0])

            # Termination condition check
            k = max(10, it // 10)
            if it > k and phis[it] < 0 and \
                    (phis[it] - phis[it - k]) / phis[it] < k * epsilon:
                break
            elif it > max_iter:
                break

            # Save progress
            out_str = 'CG: {:} / {:} - phi: {:.5f}, chkpt_loss: {:.5f}, lambd: {:.5f}'.format(
                it, max_iter, phis[-1], corr_losses[-1], lambd)
            with open(fname, 'a') as f:
                f.write(out_str + '\n')
            # Save progress

        # Start backtracking
        idx = 0
        for idx in range(1, len(saved_params))[::-1]:
            if corr_losses[idx] <= corr_losses[idx - 1]:
                chosen_ind = idx
                break

        # Return the next initialization
        with open(fname, 'a') as f:
            f.write(f'Final batch loss: {corr_losses[idx]}\n')
            f.write('=' * 80 + '\n')
            f.flush()

        return saved_params[chosen_ind], x

    def init(
            params, xi=0.5, lambd=1.0, alpha=0.75, max_iter=5,
            line_search=True, fname='cg.txt', precond='uncentered',
            use_momentum=True):
        """Initializes the Hessian Free optimizer.

        Arguments:
            params -- Parameters of the network
            xi -- Initial value of the CG information sharing coefficient
            lambd -- Initial damping strength
            alpha -- Power of the preconditioner
            max_iter -- Maximum number of CG iterations
            line_search -- Whether to use line search before returning updates
            fname -- File name to print out CG iteration information
            precond -- Preconditioner, one of "centered", "uncentered", or "none"
            use_momentum -- Whether to use the information sharing between CG runs
        """

        with open(fname, 'w') as f:
            f.write('Starting experiment\n')
            f.write('=' * 80 + '\n')

        assert precond in ['centered', 'uncentered', 'none']

        return {
            'x0': zero_vec(params),
            'lambda': np.array(lambd, dtype=np.float64),
            'alpha': np.array(alpha, dtype=np.float64),
            'xi': np.array(xi, dtype=np.float64),
            'v': zero_vec(params),
            'max_iter': max_iter,
            'line_search': line_search,
            'fname': fname,
            'precond': precond,
            'use_momentum': use_momentum
        }

    def update(batch_grad, opt_state, params, state, batch, labels):
        """Performs an update using HF. Here we break Optax's API by requiring
        more information.
        """

        # Prepare the preconditioner
        if opt_state['precond'] == 'none':
            Minv = one_vec(params)
        elif opt_state['precond'] == 'centered':
            Minv = Minv_factory_centered(
                params, state, batch, labels, opt_state['lambda'],
                opt_state['alpha'])
        else:
            Minv = Minv_factory_uncentered(
                batch_grad, opt_state['lambda'], opt_state['alpha'])

        # Turn off momentum accordingly
        if not opt_state['use_momentum']:
            opt_state['xi'] = np.array(0.0)

        # Call CG to compute descent direction and next initialization
        p, opt_state['x0'] = cg(
            opt_state['lambda'], scale_vec(-1.0, batch_grad),
            scale_vec(opt_state['xi'], opt_state['x0']), Minv, params, state,
            batch, labels, max_iter=opt_state['max_iter'],
            fname=opt_state['fname'])

        # Re-adjust lambda
        dq = dot(batch_grad, p) + 0.5 * dot(
            p, dampened(params, state, batch, labels, p, opt_state['lambda']))
        f_theta, state = loss(params, state, batch, labels)
        f_theta_p = loss(lin_comb(params, 1.0, p), state, batch, labels)[0]

        rho = (f_theta_p - f_theta) / dq
        if rho < 1 / 4:
            opt_state['lambda'] = opt_state['lambda'] * 100 / 99
        elif rho > 3 / 4:
            opt_state['lambda'] = opt_state['lambda'] * 99 / 100

        # Adjust "momentum"
        opt_state['xi'] = min(1.01 * opt_state['xi'], 0.99)

        if not opt_state['line_search']:
            return p, opt_state

        alpha, beta = 1.0, 0.8
        c = 1e-2
        for _ in range(10):  # Maximum 10 iterations
            d_theta = lin_comb(params, alpha, p)
            if loss(d_theta, state, batch, labels)[0] < f_theta + \
                    c * alpha * dot(batch_grad, p):
                break
            alpha *= beta

        return scale_vec(alpha, p), opt_state

    return namedtuple('HF', ['init', 'update'])(init, update)
