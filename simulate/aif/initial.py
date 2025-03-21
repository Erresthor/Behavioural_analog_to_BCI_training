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
from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.layer_trial import compute_step_posteriors
from actynf.jaxtynf.layer_learn import learn_after_trial
from actynf.jaxtynf.layer_options import get_learning_options,get_planning_options
from actynf.jaxtynf.shape_tools import to_log_space,get_vectorized_novelty
from actynf.jaxtynf.shape_tools import vectorize_weights

# Weights for the active inference model : 
from ..hmm_weights import basic_latent_model,simple_1D_model

ACTION_MODALITIES = ["position","angle","distance"]

def initial_params(hyperparameters,
                   model_options):
    Ns = model_options["_Ns"]
    Nu = model_options["_Nu"]
    No = model_options["_No"]
    
    
    initial_parameters_dict = {}
    
    if not(model_options["modality_selector"] is None):
        if ("initial" in model_options["modality_selector"]["biaises"]):
            # Cross action dimensions :
            initial_omega = hyperparameters["initial_omega"]
        else :
            initial_omega = jnp.zeros((3,))
        initial_parameters_dict["omega"] = initial_omega  
    
    # AIF HMM weights
    a0,b0,c0,d0,e0,u = {},{},{},{},{},{}
    for action_dimension in ACTION_MODALITIES:  
        if (model_options["free_parameters"] == "independent"):
            extract_params_from = hyperparameters[action_dimension]
        else :
            extract_params_from = hyperparameters      
                
        initial_parameters = {
            "N_feedback_ticks": No[action_dimension],
            "Ns_latent" : Ns[action_dimension],
            "N_actions" : Nu[action_dimension],
            "feedback_expected_std" : extract_params_from["perception_sigma"],
            "initial_transition_confidence" : 1.0,
            "initial_transition_stickiness" : extract_params_from["initial_transition_stickiness"],
            "reward_seeking" : extract_params_from["reward_seeking"]
        }
        if model_options["set_initial_transition_confidence"]:
            initial_parameters["initial_transition_confidence"] = extract_params_from["initial_transition_confidence"]
    
        a0_m,b0_m,c0_m,d0_m,e0_m,u_m = simple_1D_model(initial_parameters)
        a0[action_dimension] = a0_m
        b0[action_dimension] = b0_m
        c0[action_dimension] = c0_m
        d0[action_dimension] = d0_m
        
        e0[action_dimension] = e0_m
        if ("initial" in model_options["biaises"]):
            e0[action_dimension] = hyperparameters[action_dimension]["initial_e"]
        
        u[action_dimension] = u_m
        
    initial_parameters_dict["A"] = a0
    initial_parameters_dict["B"] = b0
    initial_parameters_dict["C"] = c0
    initial_parameters_dict["D"] = d0
    initial_parameters_dict["E"] = e0
    initial_parameters_dict["U"] = u
    
    return initial_parameters_dict


def initial_state(param_dict,model_options):
    Nu = model_options["_Nu"]
    
    # Initial agent state (beginning of each trial)
    initial_state_dict = {}
    
    initial_action = {}
    for action_dimension in ACTION_MODALITIES:
        Nu_dim = Nu[action_dimension]  
        initial_action[action_dimension] = jnp.zeros((Nu_dim,))
    initial_state_dict["previous_action"] = initial_action
    
    initial_previous_posteriors = {}
    for action_dimension in ACTION_MODALITIES:
        Nu_dim = Nu[action_dimension]  
        initial_previous_posteriors[action_dimension] = _normalize(jnp.ones_like(param_dict["D"][action_dimension][0]))[0]
    initial_state_dict["previous_posterior"] = initial_previous_posteriors
    
    
    initial_state_dict["B"] = param_dict["B"]
    initial_state_dict["E"] = param_dict["E"]
    
    if not(model_options["modality_selector"] is None):        
        initial_state_dict["omega"] = param_dict["omega"]  
    
    return initial_state_dict