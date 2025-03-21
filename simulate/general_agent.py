# Import the needed packages 
# 
# 1/ the usual suspects
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib


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

# from .default_parameters import get_default_parameters,get_default_hparams_ranges,get_default_parameter_priors
# from .encode_vector import _encode_params

# from .initial import initial_params,initial_state
# from .step import actor_step,predict
# from .update_params import update_params

from .qvalue.default_parameters import get_default_parameters as qvalue_get_default_parameters
from .qvalue.default_parameters import get_default_hparams_ranges as qvalue_get_default_hparams_ranges
from .qvalue.default_parameters import get_default_parameter_priors as qvalue_get_default_parameter_priors
from .qvalue.encode_vector import _encode_params as qvalue_encode_params
from .qvalue.initial import initial_params as qvalue_initial_params
from .qvalue.initial import initial_state as qvalue_initial_state
from .qvalue.step import actor_step as qvalue_actor_step
from .qvalue.step import predict as qvalue_predict
from .qvalue.update_params import update_params as qvalue_update_params

from .aif.default_parameters import get_default_parameters as aif_get_default_parameters
from .aif.default_parameters import get_default_hparams_ranges as aif_get_default_hparams_ranges
from .aif.default_parameters import get_default_parameter_priors as aif_get_default_parameter_priors
from .aif.encode_vector import _encode_params as aif_encode_params
from .aif.initial import initial_params as aif_initial_params
from .aif.initial import initial_state as aif_initial_state
from .aif.step import actor_step as aif_actor_step
from .aif.step import predict as aif_predict
from .aif.update_params import update_params as aif_update_params

from actynf.jaxtynf.layer_options import get_learning_options,get_planning_options

from . import aif as aif_1d_agents

# from .aif.default_parameters import get_default_parameters,get_default_hparams_ranges,get_default_parameter_priors
# from .aif.encode_vector import _encode_params
# from .aif.default_parameters import get_default_parameters,get_default_hparams_ranges,get_default_parameter_priors

ACTION_MODALITIES = ["position","angle","distance"]
FLAT_PRIOR = tfd.Normal(loc=0., scale=1e6)  # Approximate a flat prior
FLAT01_PRIOR = tfd.Uniform(low=-1e-5,high=1.0+1e-5)  # Uniform (flat) prior on [0,1]
MEDIUM_GAUSSIAN_PRIOR = tfd.Normal(loc=10.,scale =5.0)

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
            self.model_options["generalizer"]["qtable_generalize"] = (self.model_options["model_family"] == "latql")
        
        if (not "transitions_generalize" in self.model_options["generalizer"].keys()):
            self.model_options["generalizer"]["transitions_generalize"] = False
            
        if (not "qtable_generalize" in self.model_options["generalizer"].keys()):
            self.model_options["generalizer"]["qtable_generalize"] = False

        
        self.get_methods_from_package = "aif" if (self.model_options["model_family"] == "aif") else "qvalue"
        if self.get_methods_from_package =="aif":
            self.model_options["_Th"] = {mod:action_dimension_cst["Th"] for mod,action_dimension_cst in agent_static_parameters.items()}
        
                        
        self.default_hyperparameters = self.get_default_parameters()
        
        self.initial_ranges = None
        self.priors = None        
    
    def get_name(self):
        
        if self.model_options["model_family"] == "random":
            # Random agent : no parameters to set
            return "random"
        
        model_name = "i" if (self.model_options["free_parameters"] == "independent") else "m"
        model_name += "_" + self.model_options["model_family"]
        
        if self.model_options["model_family"] == "aif":
            model_name += self.model_options["efe_compute_method"]
            if self.model_options["learn_during_trials"]:
                model_name +="L"
            if self.model_options["learn_habits"]:
                model_name += "le"
            if self.model_options["set_initial_transition_confidence"]:
                model_name += "bc"
            
        else :
            if self.model_options["assymetric_learning_rate"]:
                model_name += "a"
        
        if "static" in self.model_options["biaises"] :
            model_name += "+b"
        if "initial" in self.model_options["biaises"] :
            model_name += "&b"
            
        if (self.model_options["generalizer"]["transitions_generalize"]):
            model_name += "-bgen"
        if (self.model_options["generalizer"]["qtable_generalize"]):
            if (self.model_options["model_family"] == "latql"):
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
            elif self.model_options["modality_selector"]["metric"] == "efe":
                model_name += "efe"
            else:
                model_name += "U"
            
            if "initial" in self.model_options["modality_selector"]["biaises"]:
                model_name += "&b"
            
            if self.model_options["modality_selector"]["focused_learning"]:
                model_name += "+fl"

                if self.model_options["modality_selector"]["independent_focused_learning_weights"]:
                    model_name += "a"
        return model_name
    
    def get_tags(self):
        tags = []
        
        if self.model_options["model_family"] == "random":
            # Random agent : no parameters to set
            return ["random"]
    
        tags.append("independent" if (self.model_options["free_parameters"] == "independent") else "mixed")
        tags.append(self.model_options["model_family"])
        
        
        
        if self.get_methods_from_package =="aif": 
            if self.model_options["learn_habits"]:
                tags.append("habits_learning")
            if self.model_options["learn_during_trials"]:
                tags.append("learn_during_trials")
            if self.model_options["set_initial_transition_confidence"]:
                tags.append("initial_conf")
            tags.append(self.model_options["efe_compute_method"])
            
        else :
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
        if self.get_methods_from_package == "aif" :
            return aif_get_default_parameters(self.model_options)
        else :
            return qvalue_get_default_parameters(self.model_options) 
        

    def get_random_parameters(self,rngkey,n_sets = 1):
        if self.initial_ranges is None:
            if self.get_methods_from_package == "aif" :
                range_dict = aif_get_default_hparams_ranges(self.model_options)
            else:
                range_dict = qvalue_get_default_hparams_ranges(self.model_options)
        else : 
            range_dict = self.initial_ranges
        
        rng_key_tree = random_split_like_tree(rngkey,range_dict)
        sampler = partial(uniform_sample_leaf,size=n_sets)
        initial_feature_vectors = tree_map(sampler,rng_key_tree,range_dict)
        return self.get_encoder()(initial_feature_vectors)
    
    def get_initial_ranges(self):
        if self.initial_ranges is None:
            if self.get_methods_from_package == "aif" :
                return aif_get_default_hparams_ranges(self.model_options)
            else:
                return qvalue_get_default_hparams_ranges(self.model_options)
        return self.initial_ranges
    
    def get_number_of_free_parameters(self):
        example_params = self.get_random_parameters(jr.PRNGKey(0))

        vls,_ = (jax.tree.flatten(tree_map(lambda x : x.shape[0],example_params)))    
            
        nbr,_ = (jax.tree.flatten(tree_map(lambda x : 1,example_params)))    
        
        return sum(vls) + self.get_delta_n_parameters(),sum(nbr)
    
    
    
    
    def get_delta_n_parameters(self):
        """ 
        Depending on the model family, remove free parameter counts if the values are part of a normalized vector (the last value can be computed from the others)
        """
        
        delta = 0
        if self.model_options["model_family"] == "random":
            return 0

        # Normalized vector outputs count for 1 less free parameter !
        if self.get_methods_from_package !="aif": 
            if "static" in self.model_options["biaises"] :
                delta -= 3  # T
            if "initial" in self.model_options["biaises"] :
                delta -= 3
        
        if not(self.model_options["modality_selector"] is None):
            if "initial" in self.model_options["modality_selector"]["biaises"]:
                delta -= 1
        return delta
    
    def get_priors(self,
                   beta_omega_dist=FLAT_PRIOR,
                   beta_fl_dist=FLAT_PRIOR,
                   beta_pi_dist = FLAT_PRIOR,
                   beta_q_dist=FLAT_PRIOR,
                   B_confidence_prior = FLAT_PRIOR,
                   B_stickiness_dist = FLAT_PRIOR,
                   reward_seeking_dist = MEDIUM_GAUSSIAN_PRIOR,
                   beta_biais_dist=FLAT_PRIOR):
        if self.priors is None:
            if self.get_methods_from_package == "aif" :
                get_default_parameter_priors = aif_get_default_parameter_priors
                
                return get_default_parameter_priors(self.model_options,
                                beta_omega_dist,
                                beta_fl_dist,
                                beta_pi_dist,
                                beta_q_dist,
                                B_confidence_prior,
                                B_stickiness_dist,
                                reward_seeking_dist)
                     
            else :
                get_default_parameter_priors = qvalue_get_default_parameter_priors
        
                return get_default_parameter_priors(self.model_options,
                                        beta_omega_dist,
                                        beta_fl_dist,
                                        beta_q_dist,
                                        beta_biais_dist)
        return self.priors
        

    def get_encoder(self):
        if self.get_methods_from_package == "aif" :
            _encode_params = aif_encode_params
        else :
            _encode_params = qvalue_encode_params
        return partial(_encode_params,model_options = self.model_options)       
    
    def get_all_functions(self,_hyperparameters=None):
        
        if _hyperparameters is None:
            _hyperparameters = self.default_hyperparameters

        
        if self.get_methods_from_package == "aif" : 
            planning_options, learning_options = {},{}
            for action_dimension in ACTION_MODALITIES:
                if (self.model_options["free_parameters"] == "independent"):
                    extract_params_from = _hyperparameters[action_dimension]
                else :
                    extract_params_from = _hyperparameters 


                # Planning options :
                planning_options[action_dimension] = get_planning_options(self.model_options["_Th"][action_dimension],
                                                                          self.model_options["efe_compute_method"],
                                                                          explore_remaining_paths=True,
                                                                          state_horizon=2,action_horizon=9,
                                                                          a_novel=False,b_novel=True,
                                                                          old_efe_computation=False)
                # get_planning_options(Th,planning_method = "sophisticated",
                #         state_horizon = model_parameters["N_state_branches"],
                #         action_horizon=model_parameters["N_action_branches"],
                #         explore_remaining_paths=model_parameters["explore_remaining"],
                #         a_novel=False,b_novel=False,
                #         old_efe_computation=True)
                
                
                # Learning options :
                if self.model_options["generalizer"]["transitions_generalize"]:
                    state_generalize_function = (lambda x : jnp.exp(-extract_params_from["gamma_generalize"]*x))
                else :
                    state_generalize_function = None
                
                if self.model_options["learn_habits"] :
                    learn_e=True
                    lr_e = extract_params_from["habits_learning_rate"]
                else :
                    learn_e = False
                    lr_e = 0.0
                    
                learning_options[action_dimension] = get_learning_options(learn_b=True,learn_d=True,learn_e=learn_e,
                                                    lr_b=extract_params_from["transition_learning_rate"],lr_d=1.0,lr_e=lr_e,
                                                    fr_b=extract_params_from["transition_forgetting_rate"],fr_d=0.0,fr_e = 0.0,
                                                    method="vanilla+backwards",
                                                    state_generalize_function=state_generalize_function,
                                                    action_generalize_table=None,cross_action_extrapolation_coeff=None)        
        
        
        if self.get_methods_from_package == "aif" :
            initial_params =  aif_initial_params
        else:
            initial_params = qvalue_initial_params
        func_initial_params = partial(initial_params,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        if self.get_methods_from_package == "aif" :
            initial_state =  aif_initial_state
        else:
            initial_state = qvalue_initial_state
        func_initial_state = partial(initial_state,
                                model_options = self.model_options)
        
        if self.get_methods_from_package == "aif" :
            actor_step =  partial(aif_actor_step,planning_options=planning_options,learning_options=learning_options)
        else:
            actor_step = qvalue_actor_step
        func_step = partial(actor_step,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        if self.get_methods_from_package == "aif" :
            predict =  partial(aif_predict,planning_options=planning_options,learning_options=learning_options)
        else:
            predict = qvalue_predict
        func_predict = partial(predict,
                                hyperparameters = _hyperparameters,
                                model_options = self.model_options)
        
        if self.get_methods_from_package == "aif" :
            update_params = partial(aif_update_params,learning_options=learning_options)
        else:
            update_params = qvalue_update_params
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
    
