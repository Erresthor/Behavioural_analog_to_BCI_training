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
from actynf.jaxtynf.layer_infer_state import compute_state_posterior

# Utils : 
from .models_utils import sample_dict_of_categoricals



def agent(hyperparameters,constants,focused_learning=True):
    """ 
    This agent tracks the reward expected by performing an action at a given modality through a Qtable.
    
    Actions are not selected independently across all modalities, but are instead selected by comparing their expected rewards. Actions
    not selected through this process are sampled randomly from the initial biais.
    """
    
    
    if hyperparameters is None:
        hyperparameters = {
            "beta_omega" : 0.0,
            "angle":{
                "alpha_Q":0.0,
                "beta_Q" :0.0,
            },
            "position":{
                "alpha_Q":0.0,
                "beta_Q" :0.0,
            },
            "distance":{
                "alpha_Q":0.0,
                "beta_Q" :0.0,
            }
        }
        for action_dimension,action_dimension_cst in constants.items():           
            No_dim = action_dimension_cst["N_outcomes"]
            # Ns_dim = action_dimension_cst["N_states"]
            Nu_dim = action_dimension_cst["N_actions"]
            hyperparameters[action_dimension]["initial_q"] = jnp.zeros((Nu_dim,))


    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        
        # cross dimension params
        encoded_hyperparameters["beta_omega"] = jnp.exp(_X["beta_omega"])
        
        # Action dimension related params :
        for action_dimension in ["position","angle","distance"]:
            action_dim_features = _X[action_dimension]
            encoded_hyperparameters[action_dimension] = {
                    "initial_q": jax.nn.softmax(action_dim_features["initial_q"]),
                    "alpha_Q": jax.nn.sigmoid(action_dim_features["alpha_Q"]),
                    "beta_Q" : jnp.exp(action_dim_features["beta_Q"])
                }
        
        return encoded_hyperparameters
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters is the initial choice kernel :
        initial_q_table = {}
        for action_dimension in ["position","angle","distance"]:
        # for action_dimension,action_dimension_cst in constants.items():
            initial_q_table[action_dimension] = hyperparameters[action_dimension]["initial_q"]
        
        # initial_q_table["angle"] = jax.nn.one_hot(0,(constants["angle"]["N_actions"]))
        return initial_q_table
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        # The initial state is the CK table and an initial action (easier integration with rw+ck model)
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            
        initial_omega = jnp.zeros((3,))
        return params,initial_omega,initial_action

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        qts,omegas,previous_actions = states
                
        qt_last = {}
        for action_dimension in ["position","angle","distance"]:
            qt_last[action_dimension] = qts[action_dimension][-1]
        
        return qt_last
    
    
    def actor_step(observation,state,params,rng_key):
        
        gauge_level,reward,trial_over,t = observation
        
        # The state of the agent stores the choice kernel and the last action performed : 
        q_t,last_omega,last_action = state
        
        keys = ["position","angle","distance"]
        if focused_learning:
            learning_weights = {key:omega_v for key,omega_v in zip(keys,last_omega)}
            
                    
        action_distribution,new_qt = {},{}           
        for action_dimension in ["position","angle","distance"]:
            action_dimension_hyperparams = hyperparameters[action_dimension]
            
            last_action_dim = last_action[action_dimension]
            q_t_dim = q_t[action_dimension]
            
            # Update the table now that we have the new reward !
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            if focused_learning:
                lr_dim = learning_weights[action_dimension]
            else :
                lr_dim = 1.0
            
            new_qt_dim = q_t_dim + lr_dim*action_dimension_hyperparams["alpha_Q"]*(reward - q_t_dim)*last_action_dim
            new_qt[action_dimension] = new_qt_dim
            
            
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_Q"]*new_qt_dim)
            
            
        
        # Action selection : this is no longer independent across all dimensions !
        # The agent compares the expected rewards of each (action dimension) system by weighting them against each other : 
        omega_values = jnp.stack(list(tree_map(lambda x,y : jnp.sum(x*y),action_distribution,new_qt).values()))  # This is the discriminator 
        soft_omega_values = jax.nn.softmax(hyperparameters["beta_omega"]*omega_values)
        softmaxed_omega = {key:omega_values for key,omega_values in zip(keys,soft_omega_values)}
                
        for action_dimension in ["position","angle","distance"]:
            action_dimension_hyperparams = hyperparameters[action_dimension]

            action_distribution[action_dimension] = jax.nn.softmax(softmaxed_omega[action_dimension]*action_dimension_hyperparams["beta_Q"]*new_qt[action_dimension])
        
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,soft_omega_values,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    
    def predict(data_timestep,state,params):
        """Predict the next action given a set of observations,
        as well as the previous internal states and parameters of the agent.

        Args:
            observation (_type_): _description_
            state (_type_): _description_
            params (_type_): _description_
            true_action : the actual action that was performed (for state updating purposes !)

        Returns:
            new_state : the 
            predicted_action : $P(u_t|o_t,s_{t-1},\theta)$
        """
        gauge_level,obs_bool_filter,reward,true_action,t = data_timestep     
                
        # The state of the agent stores the choice kernel and the last action performed : 
        q_t,last_omega,last_action = state
        
        keys = ["position","angle","distance"]
        if focused_learning:
            learning_weights = {key:omega_v for key,omega_v in zip(keys,last_omega)}
        
        action_distribution,new_qt = {},{}
        # for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            
        for action_dimension in ["position","angle","distance"]:
            action_dimension_hyperparams = hyperparameters[action_dimension]
            
            last_action_dim = last_action[action_dimension]
            q_t_dim = q_t[action_dimension]
            
            # Update the table now that we have the new reward !
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            if focused_learning:
                lr_dim = learning_weights[action_dimension]
            else :
                lr_dim = 1.0
            new_qt_dim = q_t_dim + lr_dim*action_dimension_hyperparams["alpha_Q"]*(reward - q_t_dim)*last_action_dim
            new_qt[action_dimension] = new_qt_dim
            
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_Q"]*new_qt_dim)
            
            
        
        # Action selection : this is no longer independent across all dimensions !
        # The agent compares the expected rewards of each (action dimension) system by weighting them against each other : 
        omega_values = jnp.stack(list(tree_map(lambda x,y : jnp.sum(x*y),action_distribution,new_qt).values()))  # This is the discriminator 
        soft_omega_values = jax.nn.softmax(hyperparameters["beta_omega"]*omega_values)
        softmaxed_omega = {key:omega_values for key,omega_values in zip(keys,soft_omega_values)}
                
        for action_dimension in ["position","angle","distance"]:
            action_dimension_hyperparams = hyperparameters[action_dimension]
            action_distribution[action_dimension] = jax.nn.softmax(softmaxed_omega[action_dimension]*action_dimension_hyperparams["beta_Q"]*new_qt[action_dimension])
        
        predicted_action = action_distribution
        
        # Here are the data we may want to report during the training : 
        other_data = softmaxed_omega
                
        return (new_qt,soft_omega_values,true_action),predicted_action,other_data   
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params
