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
    
    if model_options["model_family"] == "random":
        return hyperparameters
    
    if not(model_options["modality_selector"] is None):
        hyperparameters["beta_omega"] = 0.0
        
        if model_options["modality_selector"]["learn"]:
            hyperparameters["alpha_omega"] = 0.0

        if ("initial" in model_options["modality_selector"]["biaises"]):
            hyperparameters["initial_omega"] =  0.0

        if (model_options["modality_selector"]["focused_learning"]):
            if (model_options["modality_selector"]["independent_focused_learning_weights"]) :
                hyperparameters["beta_fl"] =  0.0
    
    def _populate_dictionnary_of_parameters_for_action_modality(_dict):
        # Fill with each action modality param :
        
        # All models rely on learning the reward given 
        # an action
        if model_options["assymetric_learning_rate"]:
            _dict["alpha_Q+"] = 0.0
            _dict["alpha_Q-"] = 0.0
        else :
            _dict["alpha_Q"] = 0.0
        _dict["beta_Q"] = 0.0
        
        if ("static" in model_options["biaises"]):
            _dict["beta_biais"] = 0.0
        
        if model_options["_track_transitions"]:
            _dict["transition_alpha"] = 0.0
        
        if model_options["model_family"] == "latql":
            _dict["perception_sigma"] = 0.001
            
        if (model_options["generalizer"]["qtable_generalize"] or model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = 1e3
        
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
            hyperparameters[mod]["initial_q"] = jnp.zeros((Nu[mod],))
    
    if ("static" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            hyperparameters[mod]["biais"] = jnp.zeros((Nu[mod],))
    return hyperparameters


def get_default_hparams_ranges(model_options):
    Nu = model_options["_Nu"]
    hyperparameters = {}
    
    if model_options["model_family"] == "random":
        return hyperparameters
    
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
        
        # All models rely on learning the reward given 
        # an action
        if model_options["assymetric_learning_rate"]:
            _dict["alpha_Q+"] = jnp.array([-10,10])
            _dict["alpha_Q-"] = jnp.array([-10,10])
        else :
            _dict["alpha_Q"] = jnp.array([-10,10])
        _dict["beta_Q"] = jnp.array([-3,3])
        
        if ("static" in model_options["biaises"]):
            _dict["beta_biais"] = jnp.array([-3,3])
        
        if model_options["_track_transitions"]:
            _dict["transition_alpha"] = jnp.array([-10,10])
            
        if model_options["model_family"] == "latql":
            _dict["perception_sigma"] = jnp.array([-3,3])
        
        if (model_options["generalizer"]["qtable_generalize"] or model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = jnp.array([-3,3])
        
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
            hyperparameters[mod]["initial_q"] = jnp.array([-10,10,Nu[mod]])
    
    if ("static" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            hyperparameters[mod]["biais"] = jnp.array([-10,10,Nu[mod]])
    return hyperparameters


def get_default_parameter_priors(model_options,
                                 beta_omega_prior,beta_fl_prior,beta_Q_prior,beta_biais_prior):    
    priors = {}
    
    if model_options["model_family"] == "random":
        return priors
    
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
        
        # All models rely on learning the reward given 
        # an action
        if model_options["assymetric_learning_rate"]:
            _dict["alpha_Q+"] = FLAT01_PRIOR
            _dict["alpha_Q-"] = FLAT01_PRIOR
        else :
            _dict["alpha_Q"] = FLAT01_PRIOR
        _dict["beta_Q"] = beta_Q_prior
        
        if ("static" in model_options["biaises"]):
            _dict["beta_biais"] = beta_biais_prior
        
        if model_options["_track_transitions"]:
            _dict["transition_alpha"] = FLAT01_PRIOR
            
        
        
        if model_options["model_family"] == "latql":
            _dict["perception_sigma"] = FLAT_PRIOR
            
        if (model_options["generalizer"]["qtable_generalize"] or model_options["generalizer"]["transitions_generalize"]):
            _dict["gamma_generalize"] = FLAT_PRIOR
        
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
            priors[mod]["initial_q"] = FLAT01_PRIOR
    
    if ("static" in model_options["biaises"]):
        for mod in ACTION_MODALITIES :
            priors[mod]["biais"] = FLAT01_PRIOR
    return priors