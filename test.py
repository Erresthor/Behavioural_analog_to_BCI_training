
# 1/ the usual suspects
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import tree_map
import optax

from functools import partial

# 2/ The Active Inference package 
import actynf
from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.jax_toolbox import random_split_like_tree

def compute_log_prob(_it_param,_it_prior_dist):
    # For the parameters
    _mapped = tree_map(lambda x,y : y.log_prob(x),_it_param,_it_prior_dist)
    
    if isinstance(_mapped,dict):
        _mapped = list(_mapped.values())
    
    _params_lp = jnp.stack(_mapped)
    return jnp.sum(_params_lp),_params_lp

def compute_log_prob(_it_param,_it_prior_dist):
    _mapped = tree_map(lambda x,y : jnp.sum(y.log_prob(x)),_it_param,_it_prior_dist)
    return jax.tree_util.tree_reduce(lambda x,y : x+y,_mapped),_mapped


if __name__ == "__main__":
    
    
    prior_dist = {
        "angle":{
            "alpha" : tfd.Uniform(low=0.0,high=1.0),
            "beta" : tfd.Normal(10.0,10.0)
        },
        "position":{
            "gamma" : tfd.MultivariateNormalFullCovariance(jnp.ones(4),2*jnp.eye(4))#tfd.Uniform(low=0,high=1)#
        }
    }
    
    prior_vals = {
        "angle":{
            "alpha" : 0.5,
            "beta" : 5.0
        },
        "position":{
            "gamma" : jnp.array([0.4,0.2,0.1,10.0])
        }
    }
    import tensorflow_probability
    print(tensorflow_probability.substrates.jax.__version__)
    print(jax.__version__)
    print(np.__version__)
    D = tfd.MultivariateNormalFullCovariance(jnp.zeros(4),jnp.eye(4))
    print(D)
    
    # # _mapped = tree_map(lambda x,y : y.log_prob(x),prior_vals,prior_dist)
    # print(compute_log_prob(prior_vals,prior_dist))
    
    
    func = lambda x : compute_log_prob(x,prior_dist)[0]
    
    # print(func(prior_vals))
    
    grad = jax.value_and_grad(func)
    
    
    print(grad(prior_vals))
