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
import tensorflow_probability.substrates.jax.distributions as tfd

from functools import partial

# 2/ The Active Inference package 
import actynf
from actynf.jaxtynf.jax_toolbox import random_split_like_tree

# Utils : 
# from .models_utils import discretize_normal_pdf,weighted_padded_roll,compute_js_controllability
from .agents_utils import uniform_sample_leaf

from .default_parameters import get_default_parameters,get_default_hparams_ranges,get_default_parameter_priors
from .encode_vector import _encode_params

from .initial import initial_params,initial_state
from .step import actor_step,predict
from .update_params import update_params

ACTION_MODALITIES = ["position","angle","distance"]
FLAT_PRIOR = tfd.Normal(loc=0., scale=1e6)  # Approximate a flat prior
FLAT01_PRIOR = tfd.Uniform(low=-1e-5,high=1.0+1e-5)  # Uniform (flat) prior on [0,1]

class Agent :
    def __init__(self,agent_options,agent_static_parameters):
        self.model_options = agent_options
        self.model_options["_track_transitions"] = ((agent_options["model_family"] == "latql")|(agent_options["model_family"] == "trw"))
        
        self.model_options["_No"] = {mod:action_dimension_cst["N_outcomes"] for mod,action_dimension_cst in agent_static_parameters.items()}
        self.model_options["_Nu"] = {mod:action_dimension_cst["N_actions"] for mod,action_dimension_cst in agent_static_parameters.items()}
        self.model_options["_Ns"] = {mod:action_dimension_cst["N_states"] for mod,action_dimension_cst in agent_static_parameters.items()}
        
        if (not "generalizer" in agent_options.keys()):
            self.model_options["generalizer"] = {}
            self.model_options["generalizer"]["transitions_generalize"] = False
            self.model_options["generalizer"]["qtable_generalize"] = False
        
        if (type(agent_options["generalizer"]) == bool):
            val = agent_options["generalizer"]
            self.model_options["generalizer"] = {}
            self.model_options["generalizer"]["transitions_generalize"] = val
            self.model_options["generalizer"]["qtable_generalize"] = val
        
        if (not "transitions_generalize" in self.model_options["generalizer"].keys()):
            self.model_options["generalizer"]["transitions_generalize"] = False
            
        if (not "qtable_generalize" in self.model_options["generalizer"].keys()):
            self.model_options["generalizer"]["qtable_generalize"] = False

        
        self.default_hyperparameters = self.get_default_parameters()
        
        self.initial_ranges = None
        self.priors = None
    
    def get_name(self):
        model_name = "i" if (self.model_options["free_parameters"] == "independent") else "m"
        model_name += "_" + self.model_options["model_family"]
        if self.model_options["assymetric_learning_rate"]:
            model_name += "a"
        if "static" in self.model_options["biaises"] :
            model_name += "+b"
        if "initial" in self.model_options["biaises"] :
            model_name += "&b"
        
        if (self.model_options["generalizer"]["transitions_generalize"]):
            model_name += "-bgen"
        if (self.model_options["generalizer"]["qtable_generalize"]):
            model_name += "-qgen"
        
        if not(self.model_options["modality_selector"] is None):
            model_name += "_"

            if self.model_options["modality_selector"]["learn"]:
                model_name += "omega"
            else :
                model_name += "direct"
            
            if  self.model_options["modality_selector"]["metric"] == "js_controll":
                model_name += "c"
            elif  self.model_options["modality_selector"]["metric"] == "q_value":
                model_name += "q"
            elif  self.model_options["modality_selector"]["metric"] == "surprisal":
                model_name += "f"
            
            if "initial" in self.model_options["modality_selector"]["biaises"]:
                model_name += "&b"
            
            if self.model_options["modality_selector"]["focused_learning"]:
                model_name += "+fl"

                if self.model_options["modality_selector"]["independent_focused_learning_weights"]:
                    model_name += "a"
        return model_name
    
    def get_tags(self):
        tags = []
        tags.append("independent" if (self.model_options["free_parameters"] == "independent") else "mixed")
        tags.append(self.model_options["model_family"])
        
        if self.model_options["assymetric_learning_rate"]:
            tags.append("assymetric")
        if "static" in self.model_options["biaises"] :
            tags.append("static_bias")
        if "initial" in self.model_options["biaises"] :
            tags.append("initial_bias")
        
        if (self.model_options["generalizer"]["transitions_generalize"]):
            tags.append("transition_generalize")
        if (self.model_options["generalizer"]["qtable_generalize"]):
           tags.append("qtable_generalize")
        
        if not(self.model_options["modality_selector"] is None):
            tags.append("selection_mechanism")

            if self.model_options["modality_selector"]["learn"]:
                tags.append("omega")
            else :
                tags.append("direct")
            
            if  self.model_options["modality_selector"]["metric"] == "js_controll":
                tags.append("controllability")
            elif  self.model_options["modality_selector"]["metric"] == "q_value":
                tags.append("q_val")
            elif  self.model_options["modality_selector"]["metric"] == "surprisal":
                tags.append("surprisal")
            
            if "initial" in self.model_options["modality_selector"]["biaises"]:
                tags.append("omega_initial_biais")
            
            if self.model_options["modality_selector"]["focused_learning"]:
                tags.append("focused_learning")

                if self.model_options["modality_selector"]["independent_focused_learning_weights"]:
                    tags.append("assymetric_fl")
        return tags
    
    def get_default_parameters(self):
        return get_default_parameters(self.model_options)    

    def get_random_parameters(self,rngkey,n_sets = 1):
        if self.initial_ranges is None:
            range_dict = get_default_hparams_ranges(self.model_options)
        else : 
            range_dict = self.initial_ranges
        
        rng_key_tree = random_split_like_tree(rngkey,range_dict)
        sampler = partial(uniform_sample_leaf,size=n_sets)
        initial_feature_vectors = tree_map(sampler,rng_key_tree,range_dict)
        return self.get_encoder()(initial_feature_vectors)
    
    def get_initial_ranges(self):
        if self.initial_ranges is None:
            return get_default_hparams_ranges(self.model_options)
        return self.initial_ranges
        
    def get_priors(self,
                   beta_omega_dist=FLAT_PRIOR,
                   beta_fl_dist=FLAT_PRIOR,
                   beta_q_dist=FLAT_PRIOR,
                   beta_biais_dist=FLAT_PRIOR):
        if self.priors is None:
            return get_default_parameter_priors(self.model_options,
                                    beta_omega_dist,
                                    beta_fl_dist,
                                    beta_q_dist,
                                    beta_biais_dist)
        return self.priors
        

    def get_encoder(self):
        return partial(_encode_params,model_options = self.model_options)       
    
    def get_all_functions(self,_hyperparameters=None):
        
        if _hyperparameters is None:
            _hyperparameters = self.default_hyperparameters
        
        
        func_initial_params = partial(initial_params,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        func_initial_state = partial(initial_state,
                                model_options = self.model_options)
        
        func_step = partial(actor_step,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        func_predict = partial(predict,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        func_update_params = partial(update_params,
                                model_options = self.model_options)
        
        return func_initial_params,func_initial_state,func_step,func_update_params,func_predict,self.get_encoder()

        


if __name__=="__main__":
    
    
    
    # agent_options = {
    #     "model_family" : "rw","latql","trw"
    #     "free_parameters" : "independent","mixed"
    #     "biaises" : ["static","initial"]
    #     "assymetric_learning_rate" : True,
    #     "modality_selector" : {
    #         "learn" : True
    #         "metric" : "js_controll","q_value","surprisal"
    #         "biaises" : ["initial"],
    #         "focused_learning" : True
    #         "independent_focused_learning_weights" : True,
    #     },
    #     "cross_state_generalize" : False,
    # }
    
    No = 19
    Ns = 5

    MODEL_CONSTANTS = {
        "position" : {
            "N_actions" : 9,
            "N_outcomes" : No,
            "N_states" : Ns
        },
        "angle" : {
            "N_actions" : 9,
            "N_outcomes" : No,
            "N_states" : Ns
        },
        "distance" : {
            "N_actions" : 4,
            "N_outcomes" : No,
            "N_states" : Ns
        },
    }
    
    
    
    
    agent_options = {
        "model_family" : "latql",
        "free_parameters" : "independent",
        "biaises" : ["static","initial"],
        "assymetric_learning_rate" : True,
        "modality_selector" : {
            "learn" : True,
            "metric" : "js_controll", #"q_value","surprisal"
            "biaises" : ["initial"],
            "focused_learning" : True,
            "independent_focused_learning_weights" : True
        },
        "cross_state_generalize" : {
            "qtable": False,
            "transitions": False,
        }
    }
    
    agent = Agent(agent_options,MODEL_CONSTANTS)
    
    print(agent.get_default_parameters())
    
    all_functions = agent.get_all_functions()
    
