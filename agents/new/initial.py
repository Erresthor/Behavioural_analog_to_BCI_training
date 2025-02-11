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
from .agents_utils import discretize_normal_pdf

ACTION_MODALITIES = ["position","angle","distance"]

def initial_params(hyperparameters,
                   model_options):
    Ns = model_options["_Ns"]
    Nu = model_options["_Nu"]
    No = model_options["_No"]
    # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
    # This usually allows for better responsiveness to the environment, but in this situation it may make the training
    # harder !               
    
    initial_parameters_dict = {}
    
    if model_options["model_family"] == "random":
        # Random agent : no parameters to set
        return initial_parameters_dict
    
    if not(model_options["modality_selector"] is None):
        if ("initial" in model_options["modality_selector"]["biaises"]):
            # Cross action dimensions :
            initial_omega = hyperparameters["initial_omega"]
        else :
            initial_omega = jnp.zeros((3,))
        initial_parameters_dict["omega"] = initial_omega  
        
        
    
    if model_options["model_family"] == "latql" :
        # Foreach action dimensions :
        initial_A,initial_D = {},{}        
            
        for action_dimension in ACTION_MODALITIES:  
            if (model_options["free_parameters"] == "independent"):
                extract_params_from = hyperparameters[action_dimension]
            else :
                extract_params_from = hyperparameters          
            No_dim = No[action_dimension]
            Ns_dim = Ns[action_dimension]
            Nu_dim = Nu[action_dimension]
            
            # The feedback is a one-dimensionnal information related to the latent state
            all_scalar_fb_values = jnp.linspace(0,1,Ns_dim)   # Assume that the bigger the index of the state, the better the feedback
            discretize_distance_normal_function = partial(discretize_normal_pdf,std=extract_params_from["perception_sigma"],num_bins = No_dim,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
            normal_mapping_dim,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
            
            initial_A[action_dimension] = normal_mapping_dim
            initial_D[action_dimension],_ = _normalize(jnp.ones((Ns_dim,)))

        initial_parameters_dict["A"] = initial_A
        initial_parameters_dict["D"] = initial_D
        
        # Transition matrix for a latent model
        initial_B = {}
        for action_dimension in ACTION_MODALITIES:            
            No_dim = No[action_dimension]
            Ns_dim = Ns[action_dimension]
            Nu_dim = Nu[action_dimension]            
            
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
        initial_parameters_dict["B"] = initial_B

    elif model_options["_track_transitions"] :
        # Transition matrix for a direct model : no state dimension
        
        initial_B = {}
        for action_dimension in ACTION_MODALITIES:            
            No_dim = No[action_dimension]
            Nu_dim = Nu[action_dimension]            
            
            initial_B[action_dimension],_ = _normalize(jnp.ones((No_dim,No_dim,Nu_dim)),axis=0)
        initial_parameters_dict["B"] = initial_B
    

    initial_q_table = {}
    for action_dimension in ACTION_MODALITIES:    
        Nu_dim = Nu[action_dimension]     
                             
        if ("initial" in model_options["biaises"]):
            prior_q_table = hyperparameters[action_dimension]["initial_q"]
        else : 
            prior_q_table = jnp.zeros((Nu_dim,))
        
        
        if model_options["model_family"] == "latql" :
            prior_q_table = jnp.repeat(jnp.expand_dims(prior_q_table,-1),Ns_dim,-1)
        
        initial_q_table[action_dimension] = prior_q_table
    initial_parameters_dict["q_table"] = initial_q_table
    
    return initial_parameters_dict


def initial_state(param_dict,model_options):
    Nu = model_options["_Nu"]
    No = model_options["_No"]
    # Initial agent state (beginning of each trial)
    
    initial_state_dict = {}
    
    initial_action = {}
    for action_dimension in ACTION_MODALITIES:
        Nu_dim = Nu[action_dimension]  
        initial_action[action_dimension] = jnp.zeros((Nu_dim,))
    initial_state_dict["previous_action"] = initial_action
    
    
    if model_options["model_family"] == "random":
        # Random agent : no parameters to set except for the last action performed
        return initial_state_dict
    
    
    
    # All models use the modality-wise q_table as well as the previous action
    initial_state_dict["q_table"] = param_dict["q_table"]
    
    
    
    if model_options["model_family"] == "latql" :
        initial_previous_posteriors = {}
        for action_dimension in ACTION_MODALITIES:
            Nu_dim = Nu[action_dimension]  
            initial_previous_posteriors[action_dimension] = _normalize(jnp.ones_like(param_dict["D"][action_dimension]))[0]
        initial_state_dict["previous_posterior"] = initial_previous_posteriors
    
    if model_options["model_family"] == "tracking_rw" :
        initial_state_dict["previous_observation"] = jnp.zeros((No[ACTION_MODALITIES[0]],))
    
    if model_options["_track_transitions"] :
        initial_state_dict["B"] = param_dict["B"]
        
    if not(model_options["modality_selector"] is None):        
        initial_state_dict["omega"] = param_dict["omega"]  
    
    return initial_state_dict