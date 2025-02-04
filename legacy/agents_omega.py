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
from simulate.models_utils import discretize_normal_pdf


# This file has simple agents that perform action along 3 dims at the same time ! 
# This is useful to study action selection and learning rules in an ambiguous environment.

# All agents implement 6 classes :
#  - initial_params (no args) : initialize the parameters of the model at the beginning of the training
#  - initial_state (args : params) : initialize the inner state of the model at the beginning of the trial
#  - actor_step (args : observation,model_state,params,jax.random.PRNGKey) : used in forward mode : generates a set of action variables and updates the agent inner state in response to stimuli
#  - update_params (args : trial_history, params): change the parameters at the end of a trial, given a history of observations & inner states, u^date the trial scale parameters of the models
#  - predict (used in log likelihood computations) : the same as actor_step, without the action selection part
#  - encode_params (used in inversions) : transform a tensor of real-valued weights into a set of model parameters.


# These models are similar to the ones shown in the sibling files, with the 
# inclusion of a controlability estimation scheme as in 
# Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
# (details in https://static-content.springer.com/esm/art%3A10.1038%2Fs41562-022-01306-w/MediaObjects/41562_2022_1306_MOESM1_ESM.pdf)
# In practice, it means that model states are complemented with an omega dynamic estimator of controllability
# as well as a "spectator" transition model that does not take the action into account. (as well as related model parameters)
# These models select actions depending on the relative perceived controllability of each modality

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
                "transition_alpha" : 0.0,
                "spectator_alpha" : 0.0,
                "omega_alpha" : 0.0,
                "omega_beta" : 0.0
            },
            "position":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "spectator_alpha" : 0.0,
                "omega_alpha" : 0.0,
                "omega_beta" : 0.0
            },
            "distance":{
                "alpha_+":0.0,
                "alpha_-":0.0,
                "beta" :0.0,
                "alpha_ck" : 0.0,
                "beta_ck" : 0.0,
                "transition_alpha" : 0.0,
                "spectator_alpha" : 0.0,
                "omega_alpha" : 0.0,
                "omega_beta" : 0.0
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
        initial_q_table,initial_choice_kernel, initial_omega = {},{},{}
        initial_B,initial_Bss = {},{}
        for action_dimension,action_dimension_cst in constants.items():
            Ns_dim = action_dimension_cst["N_outcomes"]
            Nu_dim = action_dimension_cst["N_actions"]
            
            initial_q_table[action_dimension] = jnp.zeros((Nu_dim,Ns_dim))
            initial_choice_kernel[action_dimension] = jnp.zeros((Nu_dim,))
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
            initial_Bss[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim)),axis=0)
            initial_omega[action_dimension] = 0.0 # This should be a prior !
        return initial_q_table,initial_choice_kernel,initial_B,initial_Bss,initial_omega
    
    
    def initial_state(params):
        
        q,ck,B,Bss,omega = params
        initial_action = {}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            
        # The initial stimuli is empty but common across all action modalities
        initial_stimuli = [jnp.zeros((action_dimension_cst["N_outcomes"],))]
        
        return q,ck,B,Bss,omega,initial_action,initial_stimuli

    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,B,Bss,omega,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]  # The same across all submodels
        
        
        # Table updating :
        new_qt,new_ck,new_B,new_Bss,new_omega = {},{},{},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            B_dim = B[action_dimension]
            Bss_dim = Bss[action_dimension]            
            
            # Current state : 
            previous_state_dim = previous_gauge_level
            current_state_dim = current_gauge_level  
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Tracking state transitions as in Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # Observed transition for this action :
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Spectator model :
            spectator_alpha = action_dimension_hyperparams["spectator_alpha"]
            observed_transition = jnp.einsum("i,j->ij",current_state_dim,previous_state_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j->ij",1.0 - current_state_dim,previous_state_dim) # This density should be pushed towards 0 !
            new_Bss_dim = Bss_dim - spectator_alpha*unobserved_transition*Bss_dim # Memory loss
            new_Bss_dim = new_Bss_dim + spectator_alpha*observed_transition*(1.0-Bss_dim) # Adding new evidence       
            
            # New omega value : 
            omega_alpha = action_dimension_hyperparams["omega_alpha"]
            controllable_prediction = jnp.einsum("iju,j,u->i",new_B_dim,previous_state_dim,last_action_dim)
            uncontrollable_prediction = jnp.einsum("ij,j->i",new_Bss_dim,previous_state_dim)
            prediction_error_diff = jnp.sum(controllable_prediction*current_state_dim) - jnp.sum(uncontrollable_prediction*current_state_dim)
            new_omega_dim = omega[action_dimension] + omega_alpha*(prediction_error_diff - omega[action_dimension])
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            new_Bss[action_dimension] = new_Bss_dim
            new_omega[action_dimension] = new_omega_dim
            
        # Action selection : this is no longer independent across all dimensions !
        # The agent compares the controllability of each (action dimension) system by weighting them against each other : 
        omega_vector = jnp.array(list(new_omega.values()))
        modality_selection_invtemp = dict(list(hyperparameters.values())[0])["omega_beta"]
        softmaxed_omega = jax.nn.softmax(modality_selection_invtemp*omega_vector)
        controllability_selector = dict(zip(new_omega.keys(), softmaxed_omega.tolist()))        
        
        action_distribution = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():            
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt[action_dimension],current_gauge_level)
            value_table = action_dimension_hyperparams["beta"]*q_table_at_this_state
            habit_table = action_dimension_hyperparams["beta_ck"]*new_ck[action_dimension]
            action_distribution[action_dimension] = jax.nn.softmax(controllability_selector[action_dimension]*value_table + habit_table)
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        return (new_qt,new_ck,new_B,new_Bss,new_omega,vect_action_selected,current_stimuli),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,B,Bss,omega,previous_actions,previous_stimuli = states
        
        qt_last,ck_last,B_last,Bss_last,omega_last = {},{},{},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            ck_last[action_dimension] = cks[action_dimension][-1]
            B_last[action_dimension] = B[action_dimension][-1]
            Bss_last[action_dimension] = Bss[action_dimension][-1]
            omega_last[action_dimension] = omega[action_dimension][-1]
        return qt_last,ck_last,B_last,Bss_last,omega_last
    
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
        
        q_t,ck,B,Bss,omega,last_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]
        
        # Table updating :
        new_qt,new_ck,new_B,new_Bss,new_omega = {},{},{},{},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            B_dim = B[action_dimension]
            Bss_dim = Bss[action_dimension]            
            
            # Current state : 
            previous_state_dim = previous_gauge_level
            current_state_dim = current_gauge_level  
            
            # Qtable :
            # This is "where" the reward was observed in the table :
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("i,j->ij",last_action_dim,previous_gauge_level)
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_-"]*negative_reward_prediction_error)*previous_action_state
            
            # Choice kernel :
            ck_dim = ck[action_dimension]
            new_ck_dim = ck_dim + action_dimension_hyperparams["alpha_ck"]*(last_action_dim - ck_dim)*was_a_last_action
            
            # Tracking state transitions as in Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # Observed transition for this action :
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j,u->iju",current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - current_state_dim,previous_state_dim,last_action_dim) # This density should be pushed towards 0 !
            new_B_dim = B_dim - transition_alpha*unobserved_transition*B_dim # Memory loss
            new_B_dim = new_B_dim + transition_alpha*observed_transition*(1.0-B_dim) # Adding new evidence 
            
            # Spectator model :
            spectator_alpha = action_dimension_hyperparams["spectator_alpha"]
            observed_transition = jnp.einsum("i,j->ij",current_state_dim,previous_state_dim) # This density should be pushed towards 1 !
            unobserved_transition = jnp.einsum("i,j->ij",1.0 - current_state_dim,previous_state_dim) # This density should be pushed towards 0 !
            new_Bss_dim = Bss_dim - spectator_alpha*unobserved_transition*Bss_dim # Memory loss
            new_Bss_dim = new_Bss_dim + spectator_alpha*observed_transition*(1.0-Bss_dim) # Adding new evidence       
            
            # New omega value : 
            omega_alpha = action_dimension_hyperparams["omega_alpha"]
            controllable_prediction = jnp.einsum("iju,j,u->i",new_B_dim,previous_state_dim,last_action_dim)
            uncontrollable_prediction = jnp.einsum("ij,j->i",new_Bss_dim,previous_state_dim)
            prediction_error_diff = jnp.sum(controllable_prediction*current_state_dim) - jnp.sum(uncontrollable_prediction*current_state_dim)
            new_omega_dim = omega[action_dimension] + omega_alpha*(prediction_error_diff - omega[action_dimension])
            
            
            new_qt[action_dimension] = new_qt_dim
            new_ck[action_dimension] = new_ck_dim
            new_B[action_dimension] = new_B_dim
            new_Bss[action_dimension] = new_Bss_dim
            new_omega[action_dimension] = new_omega_dim
            
        # Action selection : this is no longer independent across all dimensions !
        # The agent compares the controllability of each (action dimension) system by weighting them against each other : 
        omega_vector = jnp.array(list(new_omega.values()))
        modality_selection_invtemp = dict(list(hyperparameters.values())[0])["omega_beta"]
        softmaxed_omega = jax.nn.softmax(modality_selection_invtemp*omega_vector)
        controllability_selector = dict(zip(new_omega.keys(), softmaxed_omega.tolist()))        
        
        action_distribution = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():            
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt[action_dimension],current_gauge_level)
            value_table = action_dimension_hyperparams["beta"]*q_table_at_this_state
            habit_table = action_dimension_hyperparams["beta_ck"]*new_ck[action_dimension]
            action_distribution[action_dimension] = jax.nn.softmax(controllability_selector[action_dimension]*value_table + habit_table)
               
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return (new_qt,new_ck,new_B,new_Bss,new_omega,true_action,current_stimuli),action_distribution,other_data
            # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params


