# The (Stochastic) Hessian Free Optimizer
See the [HF paper](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)
and the [Stochastic HF paper](https://arxiv.org/abs/1301.3641).

## Usage

```python
from proj.hf.optimizer import hf  # Import optimizer

# An hf instance requires access to the underlying network function (for hvps)
# and the loss function. The derivative of the loss function can be omitted.
clf = hk.transform_with_state(...)

def loss(...):
    ...

def dloss(...):  # This is optional
    ...

# Instantiating and initializing HF
opt = hf(clf, loss, dloss)
opt_state = opt.init(params, kwargs)

# Compute update
batch_grad = dloss(...)
updates, opt_state = opt.update(
    batch_grad, opt_state, params, state, X_batch, y_batch)
```