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
from simulate.hmm_weights import basic_latent_model,simple_1D_model


def softplus_inverse(y):
    return _jaxlog(jnp.exp(y) - 1)

# This file defines a set of models which all use the same functions
# This could (should) have been a set of children class but I don't have time to figure out how 
# jax autodifferentiation may interact with class instances :/

# All agents implement 6 classes :
#  - initial_params (no args) : initialize the parameters of the model at the beginning of the training
#  - initial_state (args : params) : initialize the inner state of the model at the beginning of the trial
#  - actor_step (args : observation,model_state,params,jax.random.PRNGKey) : used in forward mode : generates a set of action variables and updates the agent inner state in response to stimuli
#  - update_params (args : trial_history, params): change the parameters at the end of a trial, given a history of observations & inner states, u^date the trial scale parameters of the models
#  - predict (used in log likelihood computations) : the same as actor_step, without the action selection part
#  - encode_params (used in inversions) : transform a tensor of real-valued weights into a set of model parameters.

def random_agent(hyperparameters,constants):
    # a,b,c = hyperparameters
    num_actions, = constants
    
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
        action_distribution,_ = _normalize(jnp.ones((num_actions,)))
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])
        
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
        
        # OPTIONAL : Update states based on previous states, observations and parameters
        new_state = state
        
        # Compute action distribution using observation, states and parameters
        predicted_action,_ = _normalize(jnp.ones((num_actions,)))
        
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return new_state,predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def choice_kernel_agent(hyperparameters,constants):
    num_actions, = constants
    if hyperparameters is None:
        alpha,beta = 0,0
    else :
        alpha,beta = hyperparameters

    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_alpha,_X_beta = _X[0],_X[1]
    
        # Sigmoid reparameterization ensures values between 0 and 1
        encoded_alpha = jax.nn.sigmoid(_X_alpha)
        # Softplus reparameterization ensures non-negative values
        encoded_beta = jax.nn.softplus(_X_beta)
        
        return encoded_alpha,encoded_beta
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters is the initial choice kernel :
        CK_initial = jnp.zeros((num_actions,))
        
        return CK_initial # A function of the hyperparameters
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        # The initial state is the CK table and an initial action (easier integration with rw+ck model)
        return params,jnp.zeros((num_actions,))

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        # The state of the agent stores the choice kernel and the last action performed : 
        ck,last_action = state
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(last_action)  # No update if there was no last action
        new_ck = ck + alpha*(last_action - ck)*was_a_last_action
        
        action_distribution = jax.nn.softmax(beta*new_ck)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])        
        
        return (new_ck,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        cks,previous_actions = states
        
        ck_last = cks[-1]
        
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
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(last_action)  # No update if there was no last action
        new_ck = ck + alpha*(last_action - ck)*was_a_last_action
        
        predicted_action = jax.nn.softmax(beta*new_ck) 
        
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (new_ck,true_action),predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def rescorla_wagner_agent(hyperparameters,constants):
    if hyperparameters is None:
        alpha,beta = 0,0
    else :
        alpha,beta = hyperparameters
    num_actions, = constants
    
    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_learning_rate,_X_action_selection = _X[0],_X[1]
    
        # Sigmoid reparameterization ensures values between 0 and 1
        encoded_alpha = jax.nn.sigmoid(_X_learning_rate)
        # Softplus reparameterization ensures non-negative values
        encoded_beta = jax.nn.softplus(_X_action_selection)
        
        return encoded_alpha,encoded_beta
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters is the initial perceived reward :
        q_initial = jnp.zeros((num_actions,))
        
        return q_initial # A function of the hyperparameters
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        
        # The initial state is the q_table, as well as an initial action selected (None)
        return params,jnp.zeros((num_actions,))

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        q_t,previous_action = state
        
        # Update the table now that we have the new reward !
        q_tplus = q_t + alpha*(reward-q_t)*previous_action
        
        action_distribution = jax.nn.softmax(beta*q_tplus)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])       
        
        return (q_tplus,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
                
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        qts,previous_actions = states
        
        q_t_last = qts[-1]
        
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        return q_t_last
    
    
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
        
        q_t,previous_action = state
        
        # Update the table now that we have the new reward !
        q_tplus = q_t + alpha*(reward-q_t)*previous_action
        
        predicted_action = jax.nn.softmax(beta*q_tplus)
        
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (q_tplus,true_action),predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def rw_ck_agent(hyperparameters,constants):
    if hyperparameters is None:
        alpha,beta,alpha_ck,beta_ck = 0,0,0,0
    else :
        alpha,beta,alpha_ck,beta_ck = hyperparameters
    num_actions, = constants

    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_alpha,_X_beta,_X_alpha_ck,_X_beta_ck = _X[0],_X[1],_X[2],_X[3]
    
        # Sigmoid reparameterization ensures values between 0 and 1
        encoded_alpha = jax.nn.sigmoid(_X_alpha)
        encoded_alpha_ck = jax.nn.sigmoid(_X_alpha_ck)
        
        # Softplus reparameterization ensures non-negative values
        encoded_beta = jax.nn.softplus(_X_beta)
        encoded_beta_ck = jax.nn.softplus(_X_beta_ck)
        return encoded_alpha,encoded_beta,encoded_alpha_ck,encoded_beta_ck

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial perceived reward :
        q_initial = jnp.zeros((num_actions,))
        # and the initial choice kernel :
        ck_initial = jnp.zeros((num_actions,))
        
        return q_initial,ck_initial 
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        q,ck = params
        # The initial state is the q_table, as well as an initial action selected (None)
        return q,ck,jnp.zeros((num_actions,))

    def actor_step(observation,state,params,rng_key):
        gauge_level,reward,trial_over,t = observation
        
        q_t,ck,previous_action = state
        
        # Update the table now that we have the new reward !
        q_tplus = q_t + alpha*(reward-q_t)*previous_action
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(previous_action)  # No update if there was no last action
        new_ck = ck + alpha_ck*(previous_action - ck)*was_a_last_action
        
        
        action_distribution = jax.nn.softmax(beta*q_tplus + beta_ck*new_ck)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0]) 
        
        return (q_tplus,new_ck,vect_action_selected),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,previous_actions = states
        
        
        q_t_last,ck_last = qts[-1],cks[-1]

        return q_t_last,ck_last
    
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
        
        q_t,ck,previous_action = state
        
        # Update the table now that we have the new reward !
        q_tplus = q_t + alpha*(reward-q_t)*previous_action
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(previous_action)  # No update if there was no last action
        new_ck = ck + alpha_ck*(previous_action - ck)*was_a_last_action
                
        predicted_action = jax.nn.softmax(beta*q_tplus + beta_ck*new_ck)
        
        # Here are the data we may want to report during the training : 
        other_data = None
                
        return (q_tplus,new_ck,true_action),predicted_action,other_data
    # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

def q_learning_agent(hyperparameters,constants):
    if hyperparameters is None:
        alpha_plus,alpha_minus,beta,alpha_ck,beta_ck = 0,0,0,0,0
    else :
        alpha_plus,alpha_minus,beta,alpha_ck,beta_ck = hyperparameters
        
    num_actions,num_states = constants
    
    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_alpha_plus,_X_alpha_minus,_X_beta,_X_alpha_ck,_X_beta_ck = _X[0],_X[1],_X[2],_X[3],_X[4]
    
        # Sigmoid reparameterization ensures values between 0 and 1
        encoded_alpha_plus = jax.nn.sigmoid(_X_alpha_plus)
        encoded_alpha_minus = jax.nn.sigmoid(_X_alpha_minus)
        encoded_alpha_ck = jax.nn.sigmoid(_X_alpha_ck)
        
        # Softplus reparameterization ensures non-negative values
        encoded_beta = jax.nn.softplus(_X_beta)
        encoded_beta_ck = jax.nn.softplus(_X_beta_ck)
        return encoded_alpha_plus,encoded_alpha_minus,encoded_beta,encoded_alpha_ck,encoded_beta_ck

    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # Parameters are the initial q-table. As opposed to a RW agent, the mappings now depend on the states 
        # This usually allows for better responsiveness to the environment, but in this situation it may make the training
        # harder !
        q_initial = jnp.zeros((num_actions,num_states))
        # and the initial choice kernel :
        ck_initial = jnp.zeros((num_actions,))
        
        return q_initial,ck_initial 
    
    def initial_state(params):
        # Initial agent state (beginning of each trial)
        q,ck = params
        # The initial state is the q_table, as well as an initial action selected (None) and the last gauge level (None)
        return q,ck,jnp.zeros((num_actions,)),[jnp.zeros((num_states,))]

    def actor_step(observation,state,params,rng_key):
        current_stimuli,reward,trial_over,t = observation
        current_gauge_level = current_stimuli[0]
        
        q_t,ck,previous_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]
        
        # Update the table now that we have the new reward !
        # This is "where" the reward was observed in the table :
        previous_action_state = jnp.einsum("i,j->ij",previous_action,previous_gauge_level)
        
        positive_reward = jnp.clip(reward,min=0.0)
        negative_reward = jnp.clip(reward,max=0.0)
        
        positive_reward_prediction_error = positive_reward - q_t
        negative_reward_prediction_error = negative_reward - q_t
        
        q_tplus = q_t + (alpha_plus*positive_reward_prediction_error + alpha_minus*negative_reward_prediction_error)*previous_action_state
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(previous_action)  # No update if there was no last action
        new_ck = ck + alpha_ck*(previous_action - ck)*was_a_last_action
        


        # Action selection :
        q_table_at_this_state = jnp.einsum("ij,j->i",q_tplus,current_gauge_level)
        
        action_distribution = jax.nn.softmax(beta*q_table_at_this_state + beta_ck*new_ck)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])  
        
        return (q_tplus,new_ck,vect_action_selected,current_stimuli),(action_distribution,action_selected,vect_action_selected)

    def update_params(trial_history,params):
        rewards,observations,states,actions = trial_history
        
        qts,cks,previous_actions,previous_stimuli = states
        
        q_t_last,ck_last = qts[-1],cks[-1]
        
        # The params for the next step is the last choice kernel of the trial :
        # (the update already occured during the actor step !)
        return q_t_last,ck_last
    
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
        
        q_t,ck,previous_action,previous_stimuli = state
        previous_gauge_level = previous_stimuli[0]
        
        # Update the table now that we have the new reward !
        # This is "where" the reward was observed in the table : 
        previous_action_state = jnp.einsum("i,j->ij",previous_action,previous_gauge_level)
        
        positive_reward = jnp.clip(reward,min=0.0)
        negative_reward = jnp.clip(reward,max=0.0)
        
        positive_reward_prediction_error = positive_reward - q_t
        negative_reward_prediction_error = negative_reward - q_t
        
        q_tplus = q_t + (alpha_plus*positive_reward_prediction_error + alpha_minus*negative_reward_prediction_error)*previous_action_state
        
        # Update the choice kernel :
        was_a_last_action = jnp.sum(previous_action)  # No update if there was no last action
        new_ck = ck + alpha_ck*(previous_action - ck)*was_a_last_action
        

        # Action selection :
        q_table_at_this_state = jnp.einsum("ij,j->i",q_tplus,current_gauge_level)
        
        predicted_action = jax.nn.softmax(beta*q_table_at_this_state + beta_ck*new_ck)
        
        # Here are the data we may want to report during the training : 
        other_data = None
        
        return (q_tplus,new_ck,true_action,current_stimuli),predicted_action,other_data
            # ____________________________________________________________________________________________
    
    return initial_params,initial_state,actor_step,update_params,predict,encode_params




def active_inference_basic_1D(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {    
            # ----------------------------------------------------------------------------------------------------
            # Model parameters : these should interact with the model components in a differentiable manner
            "transition_concentration": 1.0,
            "transition_stickiness": 0.0,
            "transition_learning_rate" : 1.0,
            "state_interpolation_temperature" : 1000.0,
            
            "initial_state_concentration": 1.0,
            
            "feedback_expected_std" : 1.0,
            "emission_concentration" : 1.0,
            "emission_stickiness" : 100.0,
            
            "reward_seeking" : 0.0,
            
            "habits_learning_rate" : 1.0,
            
            "action_selection_temperature" : 0.0
        }
    
    a0,b0,c0,d0,e0,u = basic_latent_model({**constants, **hyperparameters})
    beta = hyperparameters["action_selection_temperature"]
    
    planning_options = get_planning_options(constants["Th"],"classic",a_novel=False,b_novel=True)
    learning_options = get_learning_options(learn_b=True,learn_e=constants["learn_e"],
                                            lr_b=hyperparameters["transition_learning_rate"],lr_e=hyperparameters["habits_learning_rate"],
                                    method="vanilla+backwards",
                                    state_generalize_function=lambda x : jnp.exp(-hyperparameters["state_interpolation_temperature"]*x),
                                    action_generalize_table=None,cross_action_extrapolation_coeff=None)
    
    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_b_base,_X_b_prior,_X_b_lr,_X_b_lambda = _X[0],_X[1],_X[2],_X[3]
        _X_d_base = _X[4]
        _X_a_base,_X_a_prior,_X_feedback_expected_std = _X[5],_X[6],_X[7]
        _X_rs = _X[8]
        _X_beta = _X[9]
        _X_e_lr = _X[10]
        
        
        # Sigmoid reparameterization ensures values between 0 + epsilon and 1
        enc_expected_feedback_std = jax.nn.sigmoid(_X_feedback_expected_std)
        
        # Softplus reparameterization ensures non-negative values
        enc_a_base = jax.nn.softplus(_X_a_base)
        enc_a_prior = jax.nn.softplus(_X_a_prior)
        
        enc_b_base = jax.nn.softplus(_X_b_base)
        enc_b_prior = jax.nn.softplus(_X_b_prior)
        enc_b_lr = jax.nn.softplus(_X_b_lr)
        enc_b_lambda = jax.nn.softplus(_X_b_lambda)
        
        enc_d_base = jax.nn.softplus(_X_d_base)
        encoded_habits_lr = jax.nn.softplus(_X_e_lr)
        
        encoded_beta = jax.nn.softplus(_X_beta)
        encoded_reward_seeking = jax.nn.softplus(_X_rs)       
        
        encoded_hyperparameters = {
            "transition_concentration": enc_b_base,
            "transition_stickiness": enc_b_prior,
            "transition_learning_rate" : enc_b_lr,
            "state_interpolation_temperature" : enc_b_lambda,
            
            "initial_state_concentration": enc_d_base,
            
            "feedback_expected_std" : enc_expected_feedback_std,
            "emission_concentration" : enc_a_base,
            "emission_stickiness" : enc_a_prior,
            
            "reward_seeking" : encoded_reward_seeking,
            "habits_learning_rate" : encoded_habits_lr,
            
            "action_selection_temperature" : encoded_beta,
        }
        
        return encoded_hyperparameters
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # The initial parameters of the AIF agent are its model weights :
        return a0,b0,c0,d0,e0,u
    
    def initial_state(params):
        pa,pb,pc,pd,pe,u = params

        # The "states" of the active Inference agent are : 
        # 1. The vectorized parameters for this trial :
        trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,u)
        trial_c,trial_e = to_log_space(pc,pe)
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,u,compute_a_novelty=True,compute_b_novelty=True)
        
        # 2. Its priors about the next state : (given by the d matrix parameter)
        prior = trial_d
        
        return prior,jnp.zeros_like(prior),(trial_a,trial_b,trial_c,trial_e,trial_a_nov,trial_b_nov) # We don't need trial_d anymore !

    def actor_step(observation,state,params,rng_key):
        emission,reward,trial_over,t = observation
        state_prior,previous_posterior,timestep_weights = state
        
        a_norm,b_norm,c,e,a_novel,b_novel = timestep_weights
        
        end_of_trial_filter = jnp.ones((planning_options["horizon"]+2,))
        qs,F,raw_qpi,efe = compute_step_posteriors(t,state_prior,emission,a_norm,b_norm,c,e,a_novel,b_novel,
                                    end_of_trial_filter,
                                    rng_key,planning_options)       

        # Action selection :        
        action_distribution = jax.nn.softmax(beta*efe)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])  
        
        # New state prior : 
        new_prior = jnp.einsum("iju,j,u->i",b_norm,qs,vect_action_selected)
                
        # OPTIONAL : ONLINE UPDATING OF PARAMETERS 
        
        return (new_prior,qs,timestep_weights),(action_distribution,action_selected,vect_action_selected)


    def update_params(trial_history,params):
        pa,pb,pc,pd,pe,u = params
        rewards,observations,states,actions = trial_history
        priors_history,posteriors_history,_ = states   

        obs_vect_arr = [jnp.array(observations[0])]
        qs_arr = jnp.stack(posteriors_history)
        u_vect_arr = jnp.stack(actions)    
        
        # Then, we update the parameters of our HMM model at this level
        # We use the raw weights here !
        a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                                pa,pb,pc,pd,pe,u,
                                                method = learning_options["method"],
                                                learn_what = learning_options["bool"],
                                                learn_rates = learning_options["rates"],
                                                generalize_state_function=learning_options["state_generalize_function"],
                                                generalize_action_table=learning_options["action_generalize_table"],
                                                cross_action_extrapolation_coeff=learning_options["cross_action_extrapolation_coeff"],
                                                em_iter = learning_options["em_iterations"])
        
        return a_post,b_post,c_post,d_post,e_post,u
    # ____________________________________________________________________________________________
    
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
                
        state_prior,previous_posterior,timestep_weights = state
        a_norm,b_norm,c,e,a_novel,b_novel = timestep_weights
        
        end_of_trial_filter = jnp.ones((planning_options["horizon"]+2,))
        qs,F,raw_qpi,efe = compute_step_posteriors(t,state_prior,current_stimuli,a_norm,b_norm,c,e,
                                                   a_novel,b_novel,
                                    end_of_trial_filter,
                                    None,planning_options)       

        # Action selection :        
        predicted_action = jax.nn.softmax(beta*efe)
        
        # New state prior : 
        new_prior = jnp.einsum("iju,j,u->i",b_norm,qs,true_action)
        
        # OPTIONAL : ONLINE UPDATING OF PARAMETERS 
        
        # TODO here ! (especially useful for initial trials)        
        
        # Here are the data we may want to report during the training : 
        other_data = (qs,F)
        
        return (new_prior,qs,timestep_weights),predicted_action,other_data
        # ____________________________________________________________________________________________         
    return initial_params,initial_state,actor_step,update_params,predict,encode_params






def simple_aif_1D(hyperparameters,constants):
    if hyperparameters is None:
        hyperparameters = {    
            # ----------------------------------------------------------------------------------------------------
            # Model parameters : these should interact with the model components in a differentiable manner
            "transition_stickiness": 0.0,
            "transition_learning_rate" : 1.0,
            "state_interpolation_temperature" : 1.0,
                        
            "feedback_expected_std" : 1.0,
            
            "reward_seeking" : 0.0,
            
            "habits_learning_rate" : 1.0,
            
            "action_selection_temperature" : 0.0
        }
    
    a0,b0,c0,d0,e0,u = simple_1D_model({**constants, **hyperparameters})
    beta = hyperparameters["action_selection_temperature"]
    
    planning_options = get_planning_options(constants["Th"],"classic",a_novel=False,b_novel=True)
    learning_options = get_learning_options(learn_b=True,learn_e=constants["learn_e"],
                                            lr_b=hyperparameters["transition_learning_rate"],
                                            lr_e=hyperparameters["habits_learning_rate"],
                                            method="vanilla+backwards",
                                            state_generalize_function=lambda x : jnp.exp(-hyperparameters["state_interpolation_temperature"]*x),
                                            action_generalize_table=None,cross_action_extrapolation_coeff=None)
    
    def encode_params(_X):
        # Reparametrize a real-valued vector to get the parameters of this model. Used in inversion pipelines.
        _X_b_prior,_X_b_lr,_X_b_lambda = _X[0],_X[1],_X[2]
        _X_feedback_expected_std = _X[3]
        _X_rs = _X[8]
        _X_beta = _X[9]
        _X_e_lr = _X[10]
        
        
        # Sigmoid reparameterization ensures values between 0 + epsilon and 1
        enc_expected_feedback_std = jax.nn.sigmoid(_X_feedback_expected_std)
        
        # Squared reparameterization ensures positive values       
        enc_b_prior = jnp.square(_X_b_prior)
        enc_b_lr = jnp.square(_X_b_lr)
        enc_b_lambda = jnp.square(_X_b_lambda)
        
        encoded_habits_lr = jnp.square(_X_e_lr)
        
        encoded_beta = jnp.square(_X_beta)
        encoded_reward_seeking = jnp.square(_X_rs)       
        
        encoded_hyperparameters = {
            "transition_stickiness": enc_b_prior,
            "transition_learning_rate" : enc_b_lr,
            "state_interpolation_temperature" : enc_b_lambda,
            
            "feedback_expected_std" : enc_expected_feedback_std,
            
            "reward_seeking" : encoded_reward_seeking,
            "habits_learning_rate" : encoded_habits_lr,
            
            "action_selection_temperature" : encoded_beta,
        }
        
        return encoded_hyperparameters
    
    # ____________________________________________________________________________________________
    # Each agent is a set of functions of the form :    
    def initial_params():
        # The initial parameters of the AIF agent are its model weights :
        return a0,b0,c0,d0,e0,u
    
    def initial_state(params):
        pa,pb,pc,pd,pe,u = params

        # The "states" of the active Inference agent are : 
        # 1. The vectorized parameters for this trial :
        trial_a,trial_b,trial_d = vectorize_weights(pa,pb,pd,u)
        trial_c,trial_e = to_log_space(pc,pe)
        trial_a_nov,trial_b_nov = get_vectorized_novelty(pa,pb,u,compute_a_novelty=True,compute_b_novelty=True)
        
        # 2. Its priors about the next state : (given by the d matrix parameter)
        prior = trial_d
        
        return prior,jnp.zeros_like(prior),(trial_a,trial_b,trial_c,trial_e,trial_a_nov,trial_b_nov) # We don't need trial_d anymore !

    def actor_step(observation,state,params,rng_key):
        emission,reward,trial_over,t = observation
        state_prior,previous_posterior,timestep_weights = state
        
        a_norm,b_norm,c,e,a_novel,b_novel = timestep_weights
        
        end_of_trial_filter = jnp.ones((planning_options["horizon"]+2,))
        qs,F,raw_qpi,efe = compute_step_posteriors(t,state_prior,emission,a_norm,b_norm,c,e,a_novel,b_novel,
                                    end_of_trial_filter,
                                    rng_key,planning_options)       

        # Action selection :        
        action_distribution = jax.nn.softmax(beta*efe)
        action_selected = jr.categorical(rng_key,_jaxlog(action_distribution))
        vect_action_selected = jax.nn.one_hot(action_selected,action_distribution.shape[0])  
        
        # New state prior : 
        new_prior = jnp.einsum("iju,j,u->i",b_norm,qs,vect_action_selected)
                
        # OPTIONAL : ONLINE UPDATING OF PARAMETERS 
        
        return (new_prior,qs,timestep_weights),(action_distribution,action_selected,vect_action_selected)


    def update_params(trial_history,params):
        pa,pb,pc,pd,pe,u = params
        rewards,observations,states,actions = trial_history
        priors_history,posteriors_history,_ = states   

        obs_vect_arr = [jnp.array(observations[0])]
        qs_arr = jnp.stack(posteriors_history)
        u_vect_arr = jnp.stack(actions)    
        
        # Then, we update the parameters of our HMM model at this level
        # We use the raw weights here !
        a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                                pa,pb,pc,pd,pe,u,
                                                method = learning_options["method"],
                                                learn_what = learning_options["bool"],
                                                learn_rates = learning_options["rates"],
                                                generalize_state_function=learning_options["state_generalize_function"],
                                                generalize_action_table=learning_options["action_generalize_table"],
                                                cross_action_extrapolation_coeff=learning_options["cross_action_extrapolation_coeff"],
                                                em_iter = learning_options["em_iterations"])
        
        return a_post,b_post,c_post,d_post,e_post,u
    # ____________________________________________________________________________________________
    
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
                
        state_prior,previous_posterior,timestep_weights = state
        a_norm,b_norm,c,e,a_novel,b_novel = timestep_weights
        
        end_of_trial_filter = jnp.ones((planning_options["horizon"]+2,))
        qs,F,raw_qpi,efe = compute_step_posteriors(t,state_prior,current_stimuli,a_norm,b_norm,c,e,
                                                   a_novel,b_novel,
                                    end_of_trial_filter,
                                    None,planning_options)       

        # Action selection :        
        predicted_action = jax.nn.softmax(beta*efe)
        
        # New state prior : 
        new_prior = jnp.einsum("iju,j,u->i",b_norm,qs,true_action)
        
        # OPTIONAL : ONLINE UPDATING OF PARAMETERS 
        # TODO here ! (especially useful for initial trials)        
        
        # Here are the data we may want to report during the training : 
        other_data = (qs,F)
        
        return (new_prior,qs,timestep_weights),predicted_action,other_data
        # ____________________________________________________________________________________________         
    return initial_params,initial_state,actor_step,update_params,predict,encode_params

