
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
from actynf.jaxtynf.layer_infer_state import compute_state_posterior

# Utils : 
from agents.models_utils import discretize_normal_pdf,weighted_padded_roll,compute_js_controllability
from agents.models_utils import sample_dict_of_categoricals

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
    
    # prior_vals = {
    #     "angle":{
    #         "alpha" : 0.5,
    #         "beta" : 5.0
    #     },
    #     "position":{
    #         "gamma" : jnp.array([0.4,0.2,0.1,0.3])
    #     }
    # }
    # import tensorflow_probability
    # print(tensorflow_probability.substrates.jax.__version__)
    # print(jax.__version__)
    # print(np.__version__)
    # D = tfd.MultivariateNormalFullCovariance(jnp.zeros(4),jnp.eye(4))
    # print(D)
    
    # # # _mapped = tree_map(lambda x,y : y.log_prob(x),prior_vals,prior_dist)
    # # print(compute_log_prob(prior_vals,prior_dist))
    
    
    # func = lambda x : compute_log_prob(x,prior_dist)[0]
    
    # # print(func(prior_vals))
    
    # grad = jax.value_and_grad(func)
    
    
    # print(grad(prior_vals))
    fadeout_function = lambda x : jnp.exp(-1.0*x)   
    posterior_dim = jnp.array([0.0,0.1,0.8,0.1,0.0])
    previous_latstate_dim = jnp.array([0.0,1.0,0.0,0.0,0.0])


    fig,axs = plt.subplots(2,3,figsize=(6,4))
    
    for method in range(2):
        
        if method == 1:
            axs[method,0].set_title("Two generalizations")
        
            observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)        # These cells should be pushed towards 1
            unobserved_transition = jnp.einsum("i,j->ij",1.0-posterior_dim,previous_latstate_dim)  # These cells should be pushed towards 0
            gen_observed_transition = weighted_padded_roll(observed_transition,fadeout_function,[-1,-2])
            gen_unobserved_transition = weighted_padded_roll(unobserved_transition,fadeout_function,[-1,-2])
            
            axs[method,0].imshow(observed_transition,vmin=0,vmax=1)
            axs[method,1].imshow(gen_observed_transition,vmin=0,vmax=1)
            axs[method,2].imshow(gen_unobserved_transition,vmin=0,vmax=1)
            
            print(np.round(gen_unobserved_transition,2))
        else :
            axs[method,0].set_title("One generalization")
            
            
            raw_observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)
            gen_observed_transition = weighted_padded_roll(raw_observed_transition,fadeout_function,[-1,-2])        # These cells should be pushed towards 1
            gen_unobserved_transition = jnp.sum(gen_observed_transition,-2,keepdims=True) - gen_observed_transition # These cells should be pushed towards 0

            axs[method,0].imshow(raw_observed_transition,vmin=0,vmax=1)
            axs[method,1].imshow(gen_observed_transition,vmin=0,vmax=1)
            axs[method,2].imshow(gen_unobserved_transition,vmin=0,vmax=1)
            print(np.round(gen_unobserved_transition,2))
    fig.show()
    input()