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
from actynf.jaxtynf.jax_toolbox import _normalize
from actynf.jaxtynf.layer_infer_state import compute_state_posterior

# Utils : 
from ..agents_utils import weighted_padded_roll,compute_js_controllability
from ..agents_utils import sample_dict_of_categoricals

ACTION_MODALITIES = ["position","angle","distance"]





def compute_action_posteriors(observation,state,params,          
                              hyperparameters,model_options):
    current_stimuli,reward,t = observation
    current_gauge_level = current_stimuli[0]
    
    reporting_data = {}
    
    
    if model_options["model_family"] == "random":
        final_action_distribution = {}
        for mod in ACTION_MODALITIES:
            final_action_distribution[mod],_ = _normalize(jnp.ones((model_options["_Nu"][mod],)))
        return {},final_action_distribution,reporting_data    
    
    
    
    
    # The learning weights quantify how much the subject learns from
    # the feedback across all action modalities
    step_parameters = {}
    for mod in ACTION_MODALITIES:
        step_parameters[mod] = {}
        
        if (model_options["free_parameters"] == "independent"):
            extract_params_from = hyperparameters[mod]
        else :
            extract_params_from = hyperparameters


        # Q values :
        if model_options["assymetric_learning_rate"]:
            lr_plus = extract_params_from["alpha_Q+"]
            lr_minus = extract_params_from["alpha_Q-"]
        else :
            lr_plus = extract_params_from["alpha_Q"]
            lr_minus = extract_params_from["alpha_Q"]
        step_parameters[mod]["beta_Q"] = extract_params_from["beta_Q"]
        
        # Learning rates
        step_parameters[mod]["lr+"] = lr_plus
        step_parameters[mod]["lr-"] = lr_minus
        
        # Transition tracking
        if model_options["_track_transitions"]:
            step_parameters[mod]["transition_alpha"] = extract_params_from["transition_alpha"]
        
        # Cross state generalization
        if model_options["generalizer"]["transitions_generalize"]:
            # step_parameters[mod]["gamma_generalize"] = extract_params_from["gamma_generalize"]
            step_parameters[mod]["fadeout_function_b"]  = (lambda x : jnp.exp(-extract_params_from["gamma_generalize"]*x))

        if model_options["generalizer"]["qtable_generalize"]:
            # step_parameters[mod]["gamma_generalize"] = extract_params_from["gamma_generalize"]
            step_parameters[mod]["fadeout_function_qtable"]  = (lambda x : jnp.exp(-extract_params_from["gamma_generalize"]*x))
            
            
        
        # Static biaises: 
        if ("static" in model_options["biaises"]):
            step_parameters[mod]["beta_biais"] = extract_params_from["beta_biais"]
            
    
    if not(model_options["modality_selector"] is None):
        omega = state["omega"]
        
        if (model_options["modality_selector"]["focused_learning"]):            
            if model_options["modality_selector"]["independent_focused_learning_weights"] : 
                beta_lr = hyperparameters["beta_fl"]
            else : 
                beta_lr = hyperparameters["beta_omega"]
                
            attention_learning_weights = jax.nn.softmax(beta_lr*omega)
            # omega_lr_weights = dict(zip(ACTION_MODALITIES,attention_learning_weights))
            omega_lr_weights = {key:omega_values for key,omega_values in zip(ACTION_MODALITIES,attention_learning_weights)}    
            
            for mod in ACTION_MODALITIES:
                step_parameters[mod]["lr+"] = step_parameters[mod]["lr+"]*omega_lr_weights[mod]
                step_parameters[mod]["lr-"] = step_parameters[mod]["lr-"]*omega_lr_weights[mod]
        
    if not(model_options["modality_selector"] is None):
        d_omega = {}
    
    # Actual learning : 
    for mod in ACTION_MODALITIES:
        mod_parameters = step_parameters[mod]    
        
        
        last_action_mod = state["previous_action"][mod]
        there_was_a_last_action_mod = jnp.sum(last_action_mod)  
            # Should be 0.0 if there was no previous action (trial 0)
        
        
        # A/ LEARNING
        # Learning from past actions and the new observation we just received !
        
        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        # Learning transitions
        if model_options["_track_transitions"]:
            B_mod = state["B"][mod]
            b_mod_norm,_ = _normalize(B_mod)
            transition_alpha = mod_parameters["transition_alpha"]
            
            
            if model_options["model_family"] == "latql": 
                A_mod = params["A"][mod]
                D_mod = params["D"][mod]
                previous_posterior_mod = state["previous_posterior"][mod]
                
                # Prior : even given by previous action or hard set if it is the first action
                d_dim_norm,_ = _normalize(D_mod)
                no_context_prior = d_dim_norm
                with_context_prior = jnp.einsum("iju,j,u->i",b_mod_norm,previous_posterior_mod,last_action_mod)           
                prior_dim = no_context_prior*(1.0 - there_was_a_last_action_mod) + there_was_a_last_action_mod*with_context_prior
                
                # Bayesian belief update :
                posterior_mod,F_mod = compute_state_posterior(prior_dim,[current_gauge_level],[A_mod])  # Given the available observation, I believe I am here !
                
                state["previous_posterior"][mod] = posterior_mod # Update the previous posterior for next iterations
            
                before = previous_posterior_mod
                after = posterior_mod
            elif model_options["model_family"] == "trw": 
                
                before = state["previous_observation"]
                after = current_gauge_level
                state["previous_observation"] = current_gauge_level # Update the previous posterior for next iterations
                
            else: 
                raise ValueError("Unrecognized model family name : {}".format(model_options["model_family"]))
                       
            
            
            # Transition tracking :
            observed_transition = jnp.einsum("i,j->ij",after,before)
            if model_options["generalizer"]["transitions_generalize"] :
                # fadeout_function = (lambda x : jnp.exp(-mod_parameters["gamma_generalize"]*x))
                observed_transition = weighted_padded_roll(observed_transition,mod_parameters["fadeout_function_b"],[-1,-2])
                unobserved_transitions = jnp.sum(observed_transition,-2,keepdims=True) - observed_transition
                
                db = jnp.einsum("ij,u->iju",observed_transition,last_action_mod) 
                db_unobserved = jnp.einsum("ij,u->iju",unobserved_transitions,last_action_mod)
                new_B_mod = B_mod - transition_alpha*db_unobserved*B_mod + transition_alpha*db*(1.0-B_mod)      # Update
            
            else :
                # No generalization edge cases, just push values towards 0 in
                # the target density area, and push up according to the seen transition :
                db = jnp.einsum("ij,u->iju",observed_transition,last_action_mod)
                
                observation_mask = jnp.sum(observed_transition,-2,keepdims=True)
                    # FROM which states do we believe this transition occured
                tiled_mask = jnp.repeat(observation_mask,observed_transition.shape[-2],-2)
                full_mask = jnp.einsum("ij,u->iju",tiled_mask,last_action_mod)
                
                new_B_mod = B_mod*(jnp.ones_like(db) - transition_alpha*full_mask) + transition_alpha*db
                    # Classical update scheme (exponential smoothing / EMA)
            
            state["B"][mod] = new_B_mod
            
          
        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------  
        # Qtable : all agents rely on a Q table to make decisions ! 
        # This is "where" the reward was observed in the table :
        # + allowing the agent to generalize its findings to neighboring states
        q_t_dim = state["q_table"][mod]
        
        if model_options["model_family"] == "latql": 
            previous_action_state = jnp.einsum("u,j->uj",last_action_mod,previous_posterior_mod)
            if model_options["generalizer"]["qtable_generalize"] :
                # What has been learned for a state may be generalized to other states
                previous_action_state = weighted_padded_roll(previous_action_state,mod_parameters["fadeout_function_qtable"],[-1])  
        else : 
            previous_action_state = last_action_mod  # Just previous action in this case
        
        positive_reward = jnp.clip(reward,min=0.0)
        negative_reward = jnp.clip(reward,max=0.0)
        positive_reward_prediction_error = positive_reward - q_t_dim
        negative_reward_prediction_error = negative_reward - q_t_dim
        state["q_table"][mod] = q_t_dim + (mod_parameters["lr+"]*positive_reward_prediction_error + mod_parameters["lr-"]*negative_reward_prediction_error)*previous_action_state
        
        
        
        if not(model_options["modality_selector"] is None):
            # We need a way to compute an estimator of how much this modality
            # seems to control the gauge ! 
            if model_options["modality_selector"]["metric"] == "js_controll":
                d_omega[mod] = compute_js_controllability(new_B_mod) 
            
            elif model_options["modality_selector"]["metric"] == "q_value":
                # The agent compares the expected rewards of each (action dimension) system by weighting them against each other : 
                                
                # What action would we take ?
                if model_options["model_family"] == "latql":
                    q_mod = jnp.einsum("ij,j->i",state["q_table"][mod],posterior_mod)
                else : 
                    q_mod = state["q_table"][mod]   
                
                base_term = mod_parameters["beta_Q"]*q_mod
                
                if ("static" in model_options["biaises"]):
                    biais_term = mod_parameters["beta_biais"]*hyperparameters[mod]["biais"]
                else : 
                    biais_term = 0.0
                
                action_distribution_mod = jax.nn.softmax(biais_term + base_term)  
                
                # Expected reward : 
                d_omega[mod] = jnp.sum(action_distribution_mod*q_mod)
            
            elif model_options["modality_selector"]["metric"] == "surprisal":
                
                assert model_options["model_family"] == "latql", "Can't use surprisal as a metric in the absence of a probabilistic generative model."
                d_omega[mod] = F_mod

            else :
                raise NotImplementedError("Unrecognized action modality selection metric : {}".format(model_options["modality_selector"]["metric"]))
    
    
    # If we use a modality selector, action selection is not independent across dimensions
    # but governed by an "omega" parameter. 
    
    # Updating our appreciation of omega !
    if not(model_options["modality_selector"] is None):
        d_omega_vector = jnp.array(list(d_omega.values())).flatten() # Modality selector estimator for this step
        print(d_omega_vector)
        if model_options["modality_selector"]["learn"]: # Smooth update
            state["omega"] = omega + hyperparameters["alpha_omega"]*(d_omega_vector-omega)   # Update the omega vector to better match the expected controllability for each dimension
        else : # Point update
            # We only use our point estimate of omega at this specific timestep
            state["omega"] = d_omega_vector
            
        soft_omega_values = jax.nn.softmax(hyperparameters["beta_omega"]*state["omega"])
        print(soft_omega_values)
        softmaxed_omega = {key:omega_values for key,omega_values in zip(ACTION_MODALITIES,soft_omega_values)}         
    
    
    # Computing the actual action posterior : 
    final_action_distribution = {}
    for mod in ACTION_MODALITIES:
        mod_parameters = step_parameters[mod]
        
        
        # B/ PLANNING
        # What would be the next best action ?
        if model_options["model_family"] == "latql":
            q_mod = jnp.einsum("ij,j->i",state["q_table"][mod],posterior_mod)
        else : 
            q_mod = state["q_table"][mod]   
        
        base_term = mod_parameters["beta_Q"]*q_mod
        
        if ("static" in model_options["biaises"]):
            biais_term = mod_parameters["beta_biais"]*hyperparameters[mod]["biais"]
        else : 
            biais_term = 0.0
        
        if not(model_options["modality_selector"] is None):
            attention_weight = softmaxed_omega[mod]
        else :
            attention_weight = 1.0
        
        
        action_distribution_mod = jax.nn.softmax(biais_term + attention_weight*base_term)  
        
        final_action_distribution[mod] = action_distribution_mod
        
    # The only thing missing from the state is the true action !
    return state,final_action_distribution,reporting_data



def actor_step(observation,state,params,rngkey,
                   hyperparameters,
                   model_options):
    
    gauge_level,reward,t = observation
    # augmented_observation = (gauge_level,reward,None,t)
    
    new_state,action_posteriors,other_data = compute_action_posteriors(observation,state,params,hyperparameters,model_options)

    action_selected,vect_action_selected = sample_dict_of_categoricals(action_posteriors,rngkey)   
    
    # Udpate the last action with the sampled value :
    new_state["previous_action"] = vect_action_selected
    
    return new_state, (action_posteriors,action_selected,vect_action_selected)


def predict(observation,state,params,
                   hyperparameters,
                   model_options):
    
    gauge_level,obs_bool_filter,reward,true_action,t = observation  
    reduced_observation = (gauge_level,reward,t)
    
    new_state,action_posteriors,other_data = compute_action_posteriors(reduced_observation,state,params,
                                                                       hyperparameters,
                                                                       model_options)
    
    # Udpate the last action with the observed value :
    new_state["previous_action"] = true_action
    
    return new_state, action_posteriors, other_data
