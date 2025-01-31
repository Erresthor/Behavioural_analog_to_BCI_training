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
from .models_utils import discretize_normal_pdf,weighted_padded_roll
from .models_utils import sample_dict_of_categoricals



def agent(hyperparameters,constants):
    """Model to simulate agents with a latent model of the task. 
    Descisions are taken using a mix of qtable and initial biais. 
    The Qtable is progressively learnt and allow generalization between states.
    This model tracks state transitions and can be used to implement attention-like action dimension selection.

    Args:
        hyperparameters (_type_): _description_
        constants (_type_): _description_
    """
    
    if hyperparameters is None:
        hyperparameters = {
            "angle":{
                "alpha_Q+":0.0,
                "alpha_Q-":0.0,
                "beta_Q" :0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            },
            "position":{
                "alpha_Q+":0.0,
                "alpha_Q-":0.0,
                "beta_Q" :0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            },
            "distance":{
                "alpha_Q+":0.0,
                "alpha_Q-":0.0,
                "beta_Q" :0.0,
                "transition_alpha" : 0.0,
                "perception_sigma" : 0.001,
                "gamma_generalize" : 1e3
            }
        }
        for action_dimension,action_dimension_cst in constants.items():           
            Nu_dim = action_dimension_cst["N_actions"]
            hyperparameters[action_dimension]["initial_q"] = jnp.zeros((Nu_dim,))

    def encode_params(_X):
        # Reparametrize a dictionnary of features 
        # to get the parameters of this model. Used in inversion pipelines 
        # (to force the regression parameters to live in a well defined space).       
        encoded_hyperparameters = {}
        for action_dimension,action_dim_features in _X.items():
            encoded_hyperparameters[action_dimension] = {
                    "initial_q": jax.nn.softmax(action_dim_features["initial_q"]),
                    
                    "alpha_Q+": jax.nn.sigmoid(action_dim_features["alpha_Q+"]),
                    "alpha_Q-": jax.nn.sigmoid(action_dim_features["alpha_Q-"]),
                    "beta_Q" : jnp.exp(action_dim_features["beta_Q"]),
                    
                    "transition_alpha" : jax.nn.sigmoid(action_dim_features["transition_alpha"]),
                    
                    "perception_sigma" : jnp.exp(action_dim_features["perception_sigma"]),
                    
                    "gamma_generalize" : jnp.exp(action_dim_features["gamma_generalize"])
                }
        return encoded_hyperparameters

    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !               
        initial_q_table = {}
        initial_A,initial_B,initial_D = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            No_dim = action_dimension_cst["N_outcomes"]
            Ns_dim = action_dimension_cst["N_states"]
            Nu_dim = action_dimension_cst["N_actions"]
            
            # initial_q_table[action_dimension] = jnp.zeros((Nu_dim,Ns_dim))
            
            # In this method, the prior q value is parametrized instead of the biais : 
            prior_q_table = hyperparameters[action_dimension]["initial_q"]
            prior_q_state_table = jnp.repeat(jnp.expand_dims(prior_q_table,-1),Ns_dim,-1)
            initial_q_table[action_dimension] = prior_q_state_table
            
            # The feedback is a one-dimensionnal information related to the latent state
            all_scalar_fb_values = jnp.linspace(0,1,Ns_dim)   # Assume that the bigger the index of the state, the better the feedback
            discretize_distance_normal_function = partial(discretize_normal_pdf,std=hyperparameters[action_dimension]["perception_sigma"],num_bins = No_dim,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
            normal_mapping_dim,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
            
            initial_A[action_dimension] = normal_mapping_dim
            initial_B[action_dimension],_ = _normalize(jnp.ones((Ns_dim,Ns_dim,Nu_dim)),axis=0)
            initial_D[action_dimension],_ = _normalize(jnp.ones((Ns_dim,)))
        return initial_q_table,initial_A,initial_B,initial_D

    def initial_state(params):
        q,A,B,D = params
        
        initial_action,initial_latstate_priors,initial_latstate_posteriors = {},{},{}
        for action_dimension,action_dimension_cst in constants.items():
            initial_action[action_dimension] = jnp.zeros((action_dimension_cst["N_actions"],))
            initial_latstate_priors[action_dimension] = _normalize(D[action_dimension])[0]
            initial_latstate_posteriors[action_dimension] = _normalize(jnp.ones_like(D[action_dimension]))[0]
        
        return q,B,initial_action,initial_latstate_posteriors
    
    def update_params(trial_history,params):
        # Learning, but in between trials ! 
        rewards,observations,states,actions = trial_history
        
        _,A,_,D = params
        qts,B,_,_ = states # These parameters need to be passed from trial to trial
        
        qt_last,B_last = {},{}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            qt_last[action_dimension] = qts[action_dimension][-1]
            B_last[action_dimension] = B[action_dimension][-1]
        return qt_last,A,B_last,D

    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        _,A,_,D = params
        q_t,B,last_action,last_latstate_posterior = state
                
        action_distribution= {}
        new_qt,new_B = {},{}
        posteriors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            
            # ________________________________________________________________________________________________________________________________________________________________________
            # Grouping the needed variables :
            
            # This function is used in the generalizations needed below : transitions at one state may be interpreted as evidence for similar transitions from other states.
            # (again, object oriented implementation would have been so much smoother but it should be done well to not collide with jax pipelines ...)
            fadeout_function = lambda x : jnp.exp(-action_dimension_hyperparams["gamma_generalize"]*x)  
            
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            b_dim_norm,_ = _normalize(B_dim)
                        
            # ________________________________________________________________________________________________________________________________________________________________________
            # Tables updating :
            
            # Qtable : this agent relies on a state-indexed Q table to make decisions ! 
            # This is "where" the reward was observed in the table :
            # + allowing the agent to generalize its findings to neighboring states
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            gen_action_state = weighted_padded_roll(previous_action_state,fadeout_function,[-1])  # What has been learned for a state may be generalized to other states
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_Q+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_Q-"]*negative_reward_prediction_error)*gen_action_state
            # ________________________________________________________________________________________________________________________________________________________________________
            # Tick : perception, action selection and transition learning :
            
            # Prior : even given by previous action or hard set if it is the first action
            d_dim_norm,_ = _normalize(D[action_dimension])
            no_context_prior = d_dim_norm
            with_context_prior = jnp.einsum("iju,j,u->i",b_dim_norm,previous_latstate_dim,last_action_dim)           
            prior_dim = no_context_prior*(1.0 - was_a_last_action) + was_a_last_action*with_context_prior
            
            # Bayesian belief update :
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])  # Given the available observation, I believe I am here !
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_Q"]*q_table_at_this_state)            
            
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # + allowing the agent to generalize its findings to neighboring states
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)        # These cells should be pushed towards 1
            unobserved_transition = jnp.einsum("i,j->ij",1.0-posterior_dim,previous_latstate_dim)  # These cells should be pushed towards 0
            gen_db = jnp.einsum("ij,u->iju",weighted_padded_roll(observed_transition,fadeout_function,[-1,-2]),last_action_dim)                  # Generalize to neighboring cells
            gen_db_unobserved = jnp.einsum("ij,u->iju",weighted_padded_roll(unobserved_transition,fadeout_function,[-1,-2]),last_action_dim)     # Generalize to neighboring cells
            new_B_dim = B_dim - transition_alpha*gen_db_unobserved*B_dim + transition_alpha*gen_db*(1.0-B_dim)      # Update
            
            # Updating model states :
            new_qt[action_dimension] = new_qt_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
        
        action_selected,vect_action_selected = sample_dict_of_categoricals(action_distribution,rng_key)     
        
        # Wrap the element needed for the next steps :
        next_state = (new_qt,new_B,vect_action_selected,posteriors)
        action_selected_tuple = (action_distribution,action_selected,vect_action_selected)
        return next_state,action_selected_tuple
    
    
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
        
        _,A,_,D = params
        q_t,B,last_action,last_latstate_posterior = state
        
        action_distribution= {}
        new_qt,new_B = {},{}
        posteriors = {}
        for action_dimension,action_dimension_hyperparams in hyperparameters.items():
            
            # ________________________________________________________________________________________________________________________________________________________________________
            # Grouping the needed variables :
            
            # This function is used in the generalizations needed below : transitions at one state may be interpreted as evidence for similar transitions from other states.
            # (again, object oriented implementation would have been so much smoother but it should be done well to not collide with jax pipelines ...)
            fadeout_function = lambda x : jnp.exp(-action_dimension_hyperparams["gamma_generalize"]*x)  
            
            last_action_dim = last_action[action_dimension]
            was_a_last_action = jnp.sum(last_action_dim)  # No update if there was no last action
            
            previous_latstate_dim = last_latstate_posterior[action_dimension] # The perceived state last trial
            A_dim = A[action_dimension]
            B_dim = B[action_dimension]
            b_dim_norm,_ = _normalize(B_dim)
                        
            # ________________________________________________________________________________________________________________________________________________________________________
            # Tables updating :
            
            # Qtable : this agent relies on a state-indexed Q table to make decisions ! 
            # This is "where" the reward was observed in the table :
            # + allowing the agent to generalize its findings to neighboring states
            q_t_dim = q_t[action_dimension]
            previous_action_state = jnp.einsum("u,j->uj",last_action_dim,previous_latstate_dim)
            gen_action_state = weighted_padded_roll(previous_action_state,fadeout_function,[-1])  # What has been learned for a state may be generalized to other states
            
            positive_reward = jnp.clip(reward,min=0.0)
            negative_reward = jnp.clip(reward,max=0.0)
            positive_reward_prediction_error = positive_reward - q_t_dim
            negative_reward_prediction_error = negative_reward - q_t_dim
            new_qt_dim = q_t_dim + (action_dimension_hyperparams["alpha_Q+"]*positive_reward_prediction_error + action_dimension_hyperparams["alpha_Q-"]*negative_reward_prediction_error)*gen_action_state
            # ________________________________________________________________________________________________________________________________________________________________________
            # Tick : perception, action selection and transition learning :
            
            # Prior : even given by previous action or hard set if it is the first action
            d_dim_norm,_ = _normalize(D[action_dimension])
            no_context_prior = d_dim_norm
            with_context_prior = jnp.einsum("iju,j,u->i",b_dim_norm,previous_latstate_dim,last_action_dim)           
            prior_dim = no_context_prior*(1.0 - was_a_last_action) + was_a_last_action*with_context_prior
            
            # Bayesian belief update :
            posterior_dim,F = compute_state_posterior(prior_dim,[current_gauge_level],[A_dim])  # Given the available observation, I believe I am here !
            
            # Action selection :
            q_table_at_this_state = jnp.einsum("ij,j->i",new_qt_dim,posterior_dim)
            action_distribution[action_dimension] = jax.nn.softmax(action_dimension_hyperparams["beta_Q"]*q_table_at_this_state)            
            
            # Tracking state transitions inspired from Ligneul, R., Mainen, Z.F., Ly, V. et al. Stress-sensitive inference of task controllability. Nat Hum Behav 6, 812–822 (2022). https://doi.org/10.1038/s41562-022-01306-w
            # + allowing the agent to generalize its findings to neighboring states
            transition_alpha = action_dimension_hyperparams["transition_alpha"]
            observed_transition = jnp.einsum("i,j->ij",posterior_dim,previous_latstate_dim)        # These cells should be pushed towards 1
            unobserved_transition = jnp.einsum("i,j->ij",1.0-posterior_dim,previous_latstate_dim)  # These cells should be pushed towards 0
            gen_db = jnp.einsum("ij,u->iju",weighted_padded_roll(observed_transition,fadeout_function,[-1,-2]),last_action_dim)                  # Generalize to neighboring cells
            gen_db_unobserved = jnp.einsum("ij,u->iju",weighted_padded_roll(unobserved_transition,fadeout_function,[-1,-2]),last_action_dim)     # Generalize to neighboring cells
            new_B_dim = B_dim - transition_alpha*gen_db_unobserved*B_dim + transition_alpha*gen_db*(1.0-B_dim)      # Update
            
            # Updating model states :
            new_qt[action_dimension] = new_qt_dim
            new_B[action_dimension] = new_B_dim
            posteriors[action_dimension] = posterior_dim
        
        
        # Here are the data we may want to report during the training : 
        other_data = F
        next_state = (new_qt,new_B,true_action,posteriors)
        return next_state,action_distribution,other_data
     
    return initial_params,initial_state,actor_step,update_params,predict,encode_params








