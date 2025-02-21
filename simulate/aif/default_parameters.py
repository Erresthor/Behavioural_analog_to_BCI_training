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
import tensorflow_probability.substrates.jax.distributions as tfd

ACTION_MODALITIES = ["position","angle","distance"]
FLAT_PRIOR = tfd.Normal(loc=0., scale=1e6)  # Approximate a flat prior
FLAT01_PRIOR = tfd.Uniform(low=-1e-5,high=1.0+1e-5)  # Uniform (flat) prior on [0,1]



def get_default_parameters(model_options):
    Nu = model_options["_Nu"]
    
    
    hyperparameters = {}
    
    if not(model_options["modality_selector"] is None):
        hyperparameters["beta_omega"] = 0.0
        
        if model_options["modality_selector"]["learn"]:
            hyperparameters["alpha_omega"] = 0.0

        if ("initial" in model_options["modality_selector"]["biaises"]):
            hyperparameters["initial_omega"] =  jnp.zeros((3,))

        if (model_options["modality_selector"]["focused_learning"]):
            if (model_options["modality_selector"]["independent_focused_learning_weights"]) :
                hyperparameters["beta_fl"] =  0.0
    
    
    def _populate_dictionnary_of_parameters_for_action_modality(_dict):
        # Fill with each action modality param :
        _dict["transition_stickiness"] = 0.0
        _dict["transition_learning_rate"] = 0.0
        _dict["transition_forgetting_rate"] = 0.0
        _dict["perception_sigma"] = 0.0
        _dict["reward_seeking"] = 0.0
        _dict["beta_pi"] = 0.0
        
        if (model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = 1e3        
        
        if model_options["learn_habits"]:
            _dict["habits_learning_rate"] = 0.0
        
        return _dict
        
    
    if model_options["free_parameters"] == "independent" :
        for mod in ACTION_MODALITIES :
            hyperparameters[mod] = _populate_dictionnary_of_parameters_for_action_modality({})
    else : 
        _populate_dictionnary_of_parameters_for_action_modality(hyperparameters)
        for mod in ACTION_MODALITIES :
            hyperparameters[mod] = {}
        
    # Modality specific arguments :
    if ("initial" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            hyperparameters[mod]["initial_e"] = jnp.zeros((Nu[mod],))

    return hyperparameters


def get_default_hparams_ranges(model_options):
    hyperparameters = {}
    Nu = model_options["_Nu"]
    
    if not(model_options["modality_selector"] is None):
        hyperparameters["beta_omega"] = jnp.array([-3,3])
        
        if model_options["modality_selector"]["learn"]:
            hyperparameters["alpha_omega"] = jnp.array([-10,10])

        if ("initial" in model_options["modality_selector"]["biaises"]):
            hyperparameters["initial_omega"] =  jnp.array([-10,10,3])

        if (model_options["modality_selector"]["focused_learning"]):
            if (model_options["modality_selector"]["independent_focused_learning_weights"]) :
                hyperparameters["beta_fl"] =  jnp.array([-3,3])
    
    
    def _populate_dictionnary_of_parameters_for_action_modality(_dict):
        # Fill with each action modality param :
        _dict["transition_stickiness"] = jnp.array([-3,3])
        _dict["transition_learning_rate"] = jnp.array([-3,3])
        _dict["transition_forgetting_rate"] = jnp.array([-10,10])
        _dict["perception_sigma"] = jnp.array([-3,3])
        _dict["reward_seeking"] = jnp.array([-3,3])
        _dict["beta_pi"] = jnp.array([-3,3])
        
        if (model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = jnp.array([-3,3])       
        
        if model_options["learn_habits"]:
            _dict["habits_learning_rate"] = jnp.array([-3,3])
            # No forgetting rate for habits ?
        return _dict
        
    
    if model_options["free_parameters"] == "independent" :
        for mod in ACTION_MODALITIES :
            hyperparameters[mod] = _populate_dictionnary_of_parameters_for_action_modality({})
    else : 
        _populate_dictionnary_of_parameters_for_action_modality(hyperparameters)
        for mod in ACTION_MODALITIES :
            hyperparameters[mod] = {}
        
    # Modality specific arguments :
    if ("initial" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            hyperparameters[mod]["initial_e"]= jnp.array([-3,3,Nu[mod]])
    
    return hyperparameters


def get_default_parameter_priors(model_options,
                                 beta_omega_prior,beta_fl_prior,
                                 beta_pi_prior,B_stickiness_prior,
                                 reward_seeking_prior):    
    priors = {}
    
    Nu = model_options["_Nu"]
    
    if not(model_options["modality_selector"] is None):
        priors["beta_omega"] = beta_omega_prior
        
        if model_options["modality_selector"]["learn"]:
            priors["alpha_omega"] = FLAT01_PRIOR

        if ("initial" in model_options["modality_selector"]["biaises"]):
            priors["initial_omega"] =  FLAT01_PRIOR

        if (model_options["modality_selector"]["focused_learning"]):
            if (model_options["modality_selector"]["independent_focused_learning_weights"]) :
                priors["beta_fl"] =  beta_fl_prior
    
    
    def _populate_dictionnary_of_parameters_for_action_modality(_dict):
        # Fill with each action modality param :
        _dict["transition_stickiness"] = B_stickiness_prior
        _dict["transition_learning_rate"] = FLAT_PRIOR
        _dict["transition_forgetting_rate"] = FLAT01_PRIOR
        _dict["perception_sigma"] = FLAT_PRIOR
        _dict["reward_seeking"] = reward_seeking_prior
        _dict["beta_pi"] = beta_pi_prior
        
        if (model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = FLAT_PRIOR      
        
        if model_options["learn_habits"]:
            _dict["habits_learning_rate"] = FLAT_PRIOR
        
        return _dict
        
    
    if model_options["free_parameters"] == "independent" :
        for mod in ACTION_MODALITIES :
            priors[mod] = _populate_dictionnary_of_parameters_for_action_modality({})
    else : 
        _populate_dictionnary_of_parameters_for_action_modality(priors)
        for mod in ACTION_MODALITIES :
            priors[mod] = {}
        
    # Modality specific arguments :
    if ("initial" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            priors[mod]["initial_e"]= FLAT_PRIOR
    
    return priors