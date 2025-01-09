# Import the needed packages 
# 
# 1/ the usual suspects
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map

from functools import partial

# 2/ The Active Inference package 
import actynf
from actynf.jaxtynf.jax_toolbox import _normalize

# Utils : 
from .models_utils import sample_dict_of_categoricals

# Constants should be a dictionnary of {"action dimension" : {"N_actions" : N of actions per dim}}
def agent(hyperparameters,constants):   
    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        return None
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        return None # A function of the hyperparameters
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        return None

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        # OPTIONAL : Update states based on previous states, observations and parameters
        new_state = state
        
        # Compute action distribution using observation, states and parameters
        action_distribution = {}
        for action_dimension,action_dimension_cst in constants.items():
            action_distribution[action_dimension] = _normalize(jnp.ones((action_dimension_cst["N_actions"],)))[0]
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)
        
        return new_state,(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        # Trial history is a list of trial rewards, observations and states, we may want to make them jnp arrays :
        # reward_array = jnp.stack(rewards)
        # observation_array = jnp.stack(observations)
        # states_array = jnp.stack(states)
        # action_array = jnp.stack(actions)
        
        # OPTIONAL :  Update parameters based on states, observations and actions history
        return None
    
    def predict(data_timestep,state,params):
        gauge_level,obs_bool_filter,reward,true_action,t = data_timestep
        
        # OPTIONAL : Update states based on previous states, observations and parameters
        new_state = state
        
        # Compute action distribution using observation, states and parameters
        predicted_actions = {}
        for action_dimension,action_dimension_cst in constants.items():
            predicted_actions[action_dimension] = _normalize(jnp.ones((action_dimension_cst["N_actions"],)))[0]
        
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return new_state,predicted_actions,other_data
    # ____________________________________________________________________________________________
    return initial_params,initial_state,actor_step,update_params,predict,encode_params
