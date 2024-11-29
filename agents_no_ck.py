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
from actynf.jaxtynf.layer_infer_state import compute_state_posterior


# Weights for the active inference model : 
from simulate.hmm_weights import basic_latent_model,simple_1D_model

# Utils : 
from simulate.models_utils import discretize_normal_pdf,weighted_padded_roll


# This file has simple agents that perform action along 3 dims at the same time ! 
# This is useful to study action selection and learning rules in an ambiguous environment.

# All agents implement 6 classes :
#  - initial_params (no args) : initialize the parameters of the model at the beginning of the training
#  - initial_state (args : params) : initialize the inner state of the model at the beginning of the trial
#  - actor_step (args : observation,model_state,params,jax.random.PRNGKey) : used in forward mode : generates a set of action variables and updates the agent inner state in response to stimuli
#  - update_params (args : trial_history, params): change the parameters at the end of a trial, given a history of observations & inner states, u^date the trial scale parameters of the models
#  - predict (used in log likelihood computations) : the same as actor_step, without the action selection part
#  - encode_params (used in inversions) : transform a tensor of real-valued weights into a set of model parameters.

def sample_dict_of_categoricals(dict_of_probs,rng_key):
    samples,vect_samples = {},{}
    for action_dim, probs in dict_of_probs.items():
        # Split key for each sampling operation
        rng_key, subkey = jr.split(rng_key)
        
        # Sample from the categorical distribution using each probability vector
        sample = jr.categorical(subkey, _jaxlog(probs))
        
        samples[action_dim] = sample
        vect_samples[action_dim] = jax.nn.one_hot(sample,probs.shape[0])
    
    return samples,vect_samples

# Constants should be a dictionnary of {"action dimension" : {"N_actions" : N of actions per dim}}
def random_agent(hyperparameters,constants):   
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

# Constants should be a dictionnary of {"action dimension" : {"N_actions" : N of actions per dim}}
def choice_kernel_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_ck":0.0,
                "beta_ck" :0.0,
            },
            "position":{
                "alpha_ck":0.0,
                "beta_ck" :0.0,
            },
            "distance":{
                "alpha_ck":0.0,
                "beta_ck" :0.0,
            }
        }

    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"])
                }
        return encoded_hyperparameters
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters is the initial choice kernel :
        initial_choice_kernel = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_choice_kernel[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return initial_choice_kernel
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        # The initial state is the CK table and an initial action (easier integration with rw+ck model)
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return params,initial_action

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        # The state of the agent stores the choice kernel and the last action performed : 
        ck,last_action = state
        
        action_distribution,new_ck = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            
            last_action_dim = last_action[action_dimension]
            ck_dim = ck[action_dimension]
            
            # Update the choice kernel :
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            new_ck[action_dimension] = new_ck_dim
        
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_ck,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        cks,previous_actions = states
                
        ck_last = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            ck_last[action_dimension] = cks[action_dimension][-1]
        
        return ck_last
    
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
        ck,last_action = state
        
        predicted_action,new_ck = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            ck_dim = ck[action_dimension]
            
            # Update the choice kernel :
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            predicted_action[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            new_ck[action_dimension] = new_ck_dim
            
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (new_ck,true_action),predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

# Constants should be a dictionnary of {"action dimension" : {"N_actions" : N of actions per dim}}
def rw_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha":0.0,
                "beta" :0.0,
            },
            "position":{
                "alpha":0.0,
                "beta" :0.0,
            },
            "distance":{
                "alpha":0.0,
                "beta" :0.0,
            }
        }

    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha": jax.nn.sigmoid(action_dim_features["alpha"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"])
                }
        return encoded_hyperparameters
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters is the initial choice kernel :
        initial_q_table = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_q_table[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return initial_q_table
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        # The initial state is the CK table and an initial action (easier integration with rw+ck model)
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return params,initial_action

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        # The state of the agent stores the choice kernel and the last action performed : 
        q_t,last_action = state
        
        action_distribution,new_qt = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            
            last_action_dim = last_action[action_dimension]
            q_t_dim = q_t[action_dimension]
            
            # Update the table now that we have the new reward !
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            new_qt_dim = q_t_dim + action_dimension_hyperparams["alpha"]*(reward - q_t_dim)*last_action_dim
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*new_qt_dim)
            new_qt[action_dimension] = new_qt_dim
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        qts,previous_actions = states
                
        qt_last = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
        
        return qt_last
    
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
        q_t,last_action = state
        
        action_distribution,new_qt = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            q_t_dim = q_t[action_dimension]
            
            # Update the table now that we have the new reward !
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            new_qt_dim = q_t_dim + action_dimension_hyperparams["alpha"]*(reward - q_t_dim)*last_action_dim
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*new_qt_dim)
            new_qt[action_dimension] = new_qt_dim
        predicted_action = action_distribution
        
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (new_qt,true_action),predicted_action,other_data    
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def rw_ck_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            },
            "position":{
                "alpha":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            },
            "distance":{
                "alpha":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            }
        }

    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha": jax.nn.sigmoid(action_dim_features["alpha"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"]),
                    
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"])
                }
        return encoded_hyperparameters

    def initial_params():        
        initial_q_table,initial_choice_kernel = {},{}
        for action_dimension,action_dimension_cst in constants.items():
            initial_q_table[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            initial_choice_kernel[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return initial_q_table,initial_choice_kernel
    
    def initial_state(params):
        q,ck = params
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        # The initial state is the q_table, as well as an initial action selected (None)
        return q,ck,initial_action

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        # The state of the agent stores the choice kernel and the last action performed : 
        q_t,ck,last_action = state
        
        action_distribution,new_qt,new_ck = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            
            
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            # Update the table now that we have the new reward !
            q_t_dim = q_t[action_dimension]
            new_qt_dim = q_t_dim + action_dimension_hyperparams["alpha"]*(reward - q_t_dim)*last_action_dim
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
                        
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*new_qt_dim + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
        
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,new_ck,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

        
    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,previous_actions = states
        
        qt_last,ck_last = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
        return qt_last,ck_last
    
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
        q_t,ck,last_action = state
        
        action_distribution,new_qt,new_ck = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            # Update the table now that we have the new reward !
            q_t_dim = q_t[action_dimension]
            new_qt_dim = q_t_dim + action_dimension_hyperparams["alpha"]*(reward - q_t_dim)*last_action_dim
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
                        
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*new_qt_dim + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
        predicted_action = action_distribution
        
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (new_qt,new_ck,true_action),predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def nQL_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            },
            "position":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            },
            "distance":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0
            }
        }
    
    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha_+": jax.nn.sigmoid(action_dim_features["alpha_+"]),
                    "alpha_-": jax.nn.sigmoid(action_dim_features["alpha_-"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"]),
                    
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"])
                }
        return encoded_hyperparameters

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !               
        initial_q_table,initial_choice_kernel = {},{}
        for action_dimension,action_dimension_cst in constants.items():
            initial_q_table[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],action_dimension_cst["N_outcomes"]))
            initial_choice_kernel[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        return initial_q_table,initial_choice_kernel
            
    def initial_state(params):
        
        q,ck = params
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
        # The initial state is the q_table, as well as an initial action selected (None)
        initial_stimuli = [jnp.zeros((action_dimension_cst["N_outcomes"],))]
        return q,ck,initial_action,initial_stimuli

    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]  # The same across all submodels
        
        action_distribution,new_qt,new_ck = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,current_gauge_level)
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,new_ck,vect_action_selected,current_stimuli),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,previous_actions,previous_stimuli = states
        
        qt_last,ck_last = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
        return qt_last,ck_last
    
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
        current_stimuli,obs_bool_filter,reward,true_action,t = data_timestep      
        current_gauge_level = current_stimuli[0]  
        
        q_t,ck,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]
        
        
        action_distribution,new_qt,new_ck = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,current_gauge_level)
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
                
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return (new_qt,new_ck,true_action,current_stimuli),action_distribution,other_data
            # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params


# This agent is a QLearning agent that learns interstate transitions !
def ltQL_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0
            },
            "position":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0
            },
            "distance":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0
            }
        }
    
    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha_+": jax.nn.sigmoid(action_dim_features["alpha_+"]),
                    "alpha_-": jax.nn.sigmoid(action_dim_features["alpha_-"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"]),
                    
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"]),
                    
                    "transition_alpha" : jax.nn.sigmoid(action_dim_features["transition_alpha"])
                }
        return encoded_hyperparameters

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !               
        initial_q_table,initial_choice_kernel,initial_B = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            Ns_dim = action_dimension_cst["N_outcomes"]
            Nu_dim = action_dimension_cst["N_actions"]
            
            
            
            initial_q_table[action_dimension] = jnp.zeros((Nu_dim,Ns_dim))
            initial_choice_kernel[action_dimension] = jnp.zeros((Nu_dim,))
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
        return initial_q_table,initial_choice_kernel,initial_B
            
    def initial_state(params):
        
        q,ck,B = params
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            
        # The initial stimuli is empty but common across all action modalities
        initial_stimuli = [jnp.zeros((action_dimension_cst["N_outcomes"],))]
        
        return q,ck,B,initial_action,initial_stimuli

    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,B,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]  # The same across all submodels
        
        action_distribution,new_qt,new_ck,new_B = {},{},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            B_dim = B[action_dimension]
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            # Transition table : 
            previous_state_dim = previous_gauge_level # This can be complexified in a Bayesian model
            current_state_dim = current_gauge_level
            
            # Tracking state transitions as in Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # Observed transition for this action :
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,current_gauge_level)
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,new_ck,new_B,vect_action_selected,current_stimuli),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,B,previous_actions,previous_stimuli = states
        
        qt_last,ck_last,B_last = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
            B_last[action_dimension] = B[action_dimension][-1]
        return qt_last,ck_last,B_last
    
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
        current_stimuli,obs_bool_filter,reward,true_action,t = data_timestep      
        current_gauge_level = current_stimuli[0]  
        
        q_t,ck,B,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]
        
        
        action_distribution,new_qt,new_ck,new_B = {},{},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            B_dim = B[action_dimension]
            
            # Tables updating : 
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            # Transition table : 
            previous_state_dim = previous_gauge_level # This can be complexified in a Bayesian model
            current_state_dim = current_gauge_level
            
            # Tracking state transitions as in Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # Observed transition for this action :
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,current_gauge_level)
            
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
                
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return (new_qt,new_ck,new_B,true_action,current_stimuli),action_distribution,other_data
            # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params


# This agent is a QLearning agent that learns interstate transitions !
def latQL_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
            },
            "position":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
            },
            "distance":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
            }
        }
    
    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha_+": jax.nn.sigmoid(action_dim_features["alpha_+"]),
                    "alpha_-": jax.nn.sigmoid(action_dim_features["alpha_-"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"]),
                    
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"]),
                    
                    "transition_alpha" : jax.nn.sigmoid(action_dim_features["transition_alpha"]),
                    
                    "perception_sigma" : jax.nn.softplus(action_dim_features["perception_sigma"]),
                }
        return encoded_hyperparameters

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !               
        initial_q_table,initial_choice_kernel = {},{}
        initial_A,initial_B,initial_D = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            No_dim = action_dimension_cst["N_outcomes"]
            Ns_dim = action_dimension_cst["N_states"]
            Nu_dim = action_dimension_cst["N_actions"]
            
            initial_q_table[action_dimension] = jnp.zeros((Nu_dim,Ns_dim))
            initial_choice_kernel[action_dimension] = jnp.zeros((Nu_dim,))
            
            # The feedback is a one-dimensionnal information related to the latent state
            # This is first defined using numpy, but we will derive a jax implementation if it becomes needed later
            # (i.e. we parametrize part of this mapping)
            all_scalar_fb_values = jnp.linspace(0,1,Ns_dim)   # Assume that the bigger the index of the state, the better the feedback
            discretize_distance_normal_function = partial(discretize_normal_pdf,std=hyperparameters[action_dimension]["perception_sigma"],num_bins = No_dim,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
            normal_mapping_dim,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
            
            initial_A[action_dimension] = normal_mapping_dim
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
            initial_D[action_dimension],_ = _normalize(jnp.ones((Ns_dim,)))
        return initial_q_table,initial_choice_kernel,initial_A,initial_B,initial_D
            
    def initial_state(params):
        q,ck,A,B,D = params
        
        initial_action,initial_latstate_priors,initial_latstate_posteriors = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            initial_latstate_priors[action_dimension] = D[action_dimension]
            initial_latstate_posteriors[action_dimension] = _normalize(jnp.ones_like(D[action_dimension]))[0]
        
        return q,ck,initial_latstate_priors,A,B,initial_action,initial_latstate_posteriors


    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,latstate_prior,A,B,last_action,last_latstate_posterior = state
                
        action_distribution= {}
        new_qt,new_ck,new_B = {},{},{}
        posteriors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            
            
            # Update our estimate of the current latent state:
            prior_dim = latstate_prior[action_dimension]
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])
            
            # Tables updating : 
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",posterior_dim,previous_latstate_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - posterior_dim,previous_latstate_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
            
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        
        next_priors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            # Ensure the transition tracker is normalized : (and add a prior if needed:)
            b_dim_norm,_ = _normalize(new_B[action_dimension])
            
            next_priors[action_dimension] = jnp.einsum("iju,j,u->i",b_dim_norm,posteriors[action_dimension],vect_action_selected[action_dimension])
        
        
        next_state = (new_qt,new_ck,next_priors,A,new_B,vect_action_selected,posteriors)
        action_selected_tuple = (action_distribution,action_selected,vect_action_selected)
        return next_state,action_selected_tuple

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        _,_,A,_,D = params  # These are static
        qts,cks,_,_,B,_,_ = states # These are dynamic
        
        qt_last,ck_last,B_last = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
            B_last[action_dimension] = B[action_dimension][-1]
        return qt_last,ck_last,A,B_last,D
    
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
        current_stimuli,obs_bool_filter,reward,true_action,t = data_timestep
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,latstate_prior,A,B,last_action,last_latstate_posterior = state
                
        action_distribution= {}
        new_qt,new_ck,new_B = {},{},{}
        posteriors,next_priors = {},{}
           
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            
            
            # Update our estimate of the current latent state:
            prior_dim = latstate_prior[action_dimension]
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])
            
            
            # Tables updating : 
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",posterior_dim,previous_latstate_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - posterior_dim,previous_latstate_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
            
            # Ensure the transition tracker is normalized : (and add a prior if needed:)
            b_dim_norm,_ = _normalize(new_B[action_dimension])
            
            next_priors[action_dimension] = jnp.einsum("iju,j,u->i",b_dim_norm,posteriors[action_dimension],true_action[action_dimension])
                            
        # Here are the data we may want to report during the training : 
        other_data = F
        next_state = (new_qt,new_ck,next_priors,A,new_B,true_action,posteriors)
        return next_state,action_distribution,other_data
     
    return initial_params,initial_state,actor_step,update_params,predict,encode_params



# This agent is a QLearning agent that learns interstate transitions !
def gen_latQL_agent(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            },
            "position":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            },
            "distance":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            }
        }
    
    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "alpha_+": jax.nn.sigmoid(action_dim_features["alpha_+"]),
                    "alpha_-": jax.nn.sigmoid(action_dim_features["alpha_-"]),
                    "beta" : jax.nn.softplus(action_dim_features["beta"]),
                    
                    "alpha_ck": jax.nn.sigmoid(action_dim_features["alpha_ck"]),
                    "beta_ck" : jax.nn.softplus(action_dim_features["beta_ck"]),
                    
                    "transition_alpha" : jax.nn.sigmoid(action_dim_features["transition_alpha"]),
                    
                    "perception_sigma" : jax.nn.softplus(action_dim_features["perception_sigma"]),
                    
                    "gamma_generalize" : jax.nn.softplus(action_dim_features["gamma_generalize"]),
                }
        return encoded_hyperparameters

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !               
        initial_q_table,initial_choice_kernel = {},{}
        initial_A,initial_B,initial_D = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            No_dim = action_dimension_cst["N_outcomes"]
            Ns_dim = action_dimension_cst["N_states"]
            Nu_dim = action_dimension_cst["N_actions"]
            
            initial_q_table[action_dimension] = jnp.zeros((Nu_dim,Ns_dim))
            initial_choice_kernel[action_dimension] = jnp.zeros((Nu_dim,))
            
            # The feedback is a one-dimensionnal information related to the latent state
            # This is first defined using numpy, but we will derive a jax implementation if it becomes needed later
            # (i.e. we parametrize part of this mapping)
            all_scalar_fb_values = jnp.linspace(0,1,Ns_dim)   # Assume that the bigger the index of the state, the better the feedback
            discretize_distance_normal_function = partial(discretize_normal_pdf,std=hyperparameters[action_dimension]["perception_sigma"],num_bins = No_dim,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
            normal_mapping_dim,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
            
            initial_A[action_dimension] = normal_mapping_dim
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
            initial_D[action_dimension],_ = _normalize(jnp.ones((Ns_dim,)))
        return initial_q_table,initial_choice_kernel,initial_A,initial_B,initial_D
            
    def initial_state(params):
        q,ck,A,B,D = params
        
        initial_action,initial_latstate_priors,initial_latstate_posteriors = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            initial_latstate_priors[action_dimension] = D[action_dimension]
            initial_latstate_posteriors[action_dimension] = _normalize(jnp.ones_like(D[action_dimension]))[0]
        
        return q,ck,initial_latstate_priors,A,B,initial_action,initial_latstate_posteriors


    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,latstate_prior,A,B,last_action,last_latstate_posterior = state
                
        action_distribution= {}
        new_qt,new_ck,new_B = {},{},{}
        posteriors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            
            fadeout_function = lambda x : jnp.exp(-action_dimension_hyperparams["gamma_generalize"]*x)            
            
            # Update our estimate of the current latent state:
            prior_dim = latstate_prior[action_dimension]
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])
            
            # Tables updating : 
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # + allowing the agent to generalize its findings to neighboring states
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)
            unobserved_transition = jnp.einsum("i,j->ij",1.0-posterior_dim,previous_latstate_dim)
            gen_db = jnp.einsum("ij,u->iju",weighted_padded_roll(observed_transition,fadeout_function,[-1,-2]),last_action_dim)
            gen_dbo = jnp.einsum("ij,u->iju",weighted_padded_roll(unobserved_transition,fadeout_function,[-1,-2]),last_action_dim)
            new_B_dim = B_dim - transition_alpha*gen_dbo*B_dim + transition_alpha*gen_db*(1.0-B_dim)
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            # + allowing the agent to generalize its findings to neighboring states
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            gen_action_state = weighted_padded_roll(previous_action_state,fadeout_function,[-1])
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*gen_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
            
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        
        next_priors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            # Ensure the transition tracker is normalized : (and add a prior if needed:)
            b_dim_norm,_ = _normalize(new_B[action_dimension])
            
            next_priors[action_dimension] = jnp.einsum("iju,j,u->i",b_dim_norm,posteriors[action_dimension],vect_action_selected[action_dimension])
        
        
        next_state = (new_qt,new_ck,next_priors,A,new_B,vect_action_selected,posteriors)
        action_selected_tuple = (action_distribution,action_selected,vect_action_selected)
        return next_state,action_selected_tuple

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        _,_,A,_,D = params  # These are static
        qts,cks,_,_,B,_,_ = states # These are dynamic
        
        qt_last,ck_last,B_last = {},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
            B_last[action_dimension] = B[action_dimension][-1]
        return qt_last,ck_last,A,B_last,D
    
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
        current_stimuli,obs_bool_filter,reward,true_action,t = data_timestep
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,latstate_prior,A,B,last_action,last_latstate_posterior = state
                
        action_distribution= {}
        new_qt,new_ck,new_B = {},{},{}
        posteriors,next_priors = {},{}
           
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            fadeout_function = lambda x : jnp.exp(-action_dimension_hyperparams["gamma_generalize"]*x)       
            
            
            # Update our estimate of the current latent state:
            prior_dim = latstate_prior[action_dimension]
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])
            
            
            # Tables updating : 
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # + allowing the agent to generalize its findings to neighboring states
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)
            unobserved_transition = jnp.einsum("i,j->ij",1.0-posterior_dim,previous_latstate_dim)
            gen_db = jnp.einsum("ij,u->iju",weighted_padded_roll(observed_transition,fadeout_function,[-1,-2]),last_action_dim)
            gen_dbo = jnp.einsum("ij,u->iju",weighted_padded_roll(unobserved_transition,fadeout_function,[-1,-2]),last_action_dim)
            new_B_dim = B_dim - transition_alpha*gen_dbo*B_dim + transition_alpha*gen_db*(1.0-B_dim)
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            # + allowing the agent to generalize its findings to neighboring states
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            gen_action_state = weighted_padded_roll(previous_action_state,fadeout_function,[-1])
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*gen_action_state
            
            # Update the choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta"]*q_table_at_this_state + action_dimension_hyperparams["beta_ck"]*new_ck_dim)
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
            
            # Ensure the transition tracker is normalized : (and add a prior if needed:)
            b_dim_norm,_ = _normalize(new_B[action_dimension])
            
            next_priors[action_dimension] = jnp.einsum("iju,j,u->i",b_dim_norm,posteriors[action_dimension],true_action[action_dimension])
                            
        # Here are the data we may want to report during the training : 
        other_data = F
        next_state = (new_qt,new_ck,next_priors,A,new_B,true_action,posteriors)
        return next_state,action_distribution,other_data
     
    return initial_params,initial_state,actor_step,update_params,predict,encode_params
