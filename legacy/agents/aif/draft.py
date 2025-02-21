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
from ...simulate.hmm_weights import basic_latent_model,simple_1D_model


def get_simple_aif_1D_functions(hyperparameters,constants,model_options):
    
    
    
    
    
    
    
    
    
    if hyperparameters is None:
        hyperparameters = {    
            # ----------------------------------------------------------------------------------------------------
            # Model parameters : these should interact with the model components in a differentiable manner
            "transition_stickiness": 0.0,
            
            "alpha_b" : 1.0,
            "alpha_e" : 1.0,
            
            "beta_pi" : 0.0
            
            "gamma_generalize" : 1.0,
                        
            "feedback_expected_std" : 1.0,
            
            "reward_seeking" : 0.0,
                        
            
        }
    
    a0,b0,c0,d0,e0,u = simple_1D_model({**constants, **hyperparameters})
    beta = hyperparameters["beta_pi"]
    
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

