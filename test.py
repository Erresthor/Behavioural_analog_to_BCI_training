
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
    
    
    # prior_dist = {
    #     "angle":{
    #         "alpha" : tfd.Uniform(low=0.0,high=1.0),
    #         "beta" : tfd.Normal(10.0,10.0)
    #     },
    #     "position":{
    #         "gamma" : tfd.MultivariateNormalFullCovariance(jnp.ones(4),2*jnp.eye(4))#tfd.Uniform(low=0,high=1)#
    #     }
    # }

    def arePointInScreen(Xs):
        return np.all((0 <= Xs) & (Xs <= 1))
    
    xs = np.array([
        [0.5,0.9],
        [0.2,1.2]
    ])
    
    print(arePointInScreen(xs))