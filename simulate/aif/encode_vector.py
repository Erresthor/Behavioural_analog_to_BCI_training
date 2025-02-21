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

ACTION_MODALITIES = ["position","angle","distance"]


# model_options = {
#     "model_family" : "aif",
#     "free_parameters" : "independent",
#     "biaises": ["initial"],
#     "learn_during_trials" : True,
#     "modality_selector" : {
#         "focused_learning" : True,
#         "independent_focused_learning_weights" : True,
#         "biaises" : ["initial"],
#         "learn" : True
#     },
#     "generalizer" : {
#         "transitions_generalize" : True
#     }
# }

def _encode_params(_X,model_options):
    # Reparametrize a dictionnary of features 
    # to get the parameters of this model. Used in inversion pipelines 
    # (to force the regression parameters to live in a well defined space).       
    encoded_hyperparameters = {}
    
    
    if not(model_options["modality_selector"] is None):
        encoded_hyperparameters["beta_omega"] = jnp.exp(_X["beta_omega"])
        
        if model_options["modality_selector"]["learn"]:
            encoded_hyperparameters["alpha_omega"] = jax.nn.sigmoid(_X["alpha_omega"])

        if ("initial" in model_options["modality_selector"]["biaises"]):
            encoded_hyperparameters["initial_omega"] =  jax.nn.softmax(_X["initial_omega"])*2 - 1.0

        if (model_options["modality_selector"]["focused_learning"]):
            if (model_options["modality_selector"]["independent_focused_learning_weights"]) :
                encoded_hyperparameters["beta_fl"] =  jnp.exp(_X["beta_fl"])
    
    
    def __encode_dictionnary_of_parameters_for_action_modality(__dict,__X):
        # Fill with each action modality param :
        __dict["transition_stickiness"] = jnp.exp(__X["transition_stickiness"])
        __dict["transition_learning_rate"] = jnp.exp(__X["transition_learning_rate"])
        __dict["transition_forgetting_rate"] = jax.nn.sigmoid(__X["transition_forgetting_rate"])
        __dict["perception_sigma"] = jnp.exp(__X["perception_sigma"])
        __dict["reward_seeking"] = jnp.exp(__X["reward_seeking"])
        __dict["beta_pi"] = jnp.exp(__X["beta_pi"])
        
        if (model_options["generalizer"]["transitions_generalize"]):
            __dict["gamma_generalize"] = jnp.exp(__X["gamma_generalize"])    
        
        if model_options["learn_habits"]:
            __dict["habits_learning_rate"] = jnp.exp(__X["habits_learning_rate"])
        
        return __dict
        
    
    if model_options["free_parameters"] == "independent" :
        for mod in ACTION_MODALITIES :
            encoded_hyperparameters[mod] = __encode_dictionnary_of_parameters_for_action_modality({},_X[mod])
    else : 
        __encode_dictionnary_of_parameters_for_action_modality(encoded_hyperparameters,_X)
        for mod in ACTION_MODALITIES :
            encoded_hyperparameters[mod] = {}
        
    # Modality specific arguments :
    if ("initial" in model_options["biaises"]):
        for action_dimension in ACTION_MODALITIES :
            action_dim_features = _X[action_dimension]
            encoded_hyperparameters[action_dimension]["initial_e"] = jnp.exp(action_dim_features["initial_e"])
    
    return encoded_hyperparameters
