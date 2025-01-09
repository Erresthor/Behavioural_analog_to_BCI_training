import numpy as np
import jax
import jax.numpy as jnp

def remove_by_indices(iter, idxs):
    return [e for i, e in enumerate(iter) if i not in idxs]

def to_shape(a, shape,fill_with_value=0):
    x_, = shape
    x, = a.shape
    x_pad = (x_-x)
    return np.pad(a,(x_pad//2, x_pad//2 + x_pad%2),
                  mode = 'constant',
                  constant_values=fill_with_value)
    
def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


