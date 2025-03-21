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

from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.layer_trial import compute_step_posteriors
from actynf.jaxtynf.layer_learn import learn_after_trial,learn_during_trial
from actynf.jaxtynf.layer_options import get_learning_options,get_planning_options
from actynf.jaxtynf.shape_tools import to_log_space,get_vectorized_novelty
from actynf.jaxtynf.shape_tools import vectorize_weights


ACTION_MODALITIES = ["position","angle","distance"]



def compute_action_posteriors(observation,state,params,          
                              hyperparameters,
                              planning_options,learning_options,
                              model_options):
    current_stimuli,reward,t = observation
    current_gauge_level = current_stimuli[0]
    
    reporting_data = {}
    
    
    # The learning weights quantify how much the subject learns from
    # the feedback across all action modalities
    step_parameters = {}
    for mod in ACTION_MODALITIES:
        step_parameters[mod] = {}
        
        if (model_options["free_parameters"] == "independent"):
            extract_params_from = hyperparameters[mod]
        else :
            extract_params_from = hyperparameters
        
        step_parameters[mod]["beta_pi"] = extract_params_from["beta_pi"]
        step_parameters[mod]["transition_learning_rate"] = extract_params_from["transition_learning_rate"]
        step_parameters[mod]["transition_forgetting_rate"] = extract_params_from["transition_forgetting_rate"]
        
        if model_options["learn_habits"]:
            step_parameters[mod]["habits_learning_rate"] = extract_params_from["habits_learning_rate"]
        
    
    if not(model_options["modality_selector"] is None):
        omega = state["omega"]
        
        if (model_options["modality_selector"]["focused_learning"]):
            # 
            
            # if model_options["modality_selector"]["independent_focused_learning_weights"] : 
            #     beta_lr = hyperparameters["beta_fl"]
            # else : 
            #     beta_lr = hyperparameters["beta_omega"]
                
            # attention_learning_weights = jax.nn.softmax(beta_lr*omega)
            
            # omega_lr_weights = dict(zip(ACTION_MODALITIES,attention_learning_weights))
            
            # for mod in ACTION_MODALITIES:
            #      step_parameters[mod]["transition_learning_rate"] = omega_lr_weights[mod]*step_parameters[mod]["transition_learning_rate"]
            
            raise NotImplementedError("Focused Learning in Active Inference has not been implemented yet ...")
        
        
            # TODO : focused learning should probably make the state inference much more noisy, perhaps
            # adding noise to the perceived last action ?
    
    
    efes = {}
    if not(model_options["modality_selector"] is None):
        d_omega = {}
    
    for mod in ACTION_MODALITIES:
        mod_parameters = step_parameters[mod]    
        
        last_action_mod = state["previous_action"][mod]
        there_was_a_last_action_mod = jnp.sum(last_action_mod)  
        previous_posterior_mod = state["previous_posterior"][mod]
            
            
        # The "states" of the active Inference agent are : 
        # 1. The vectorized parameters for this trial :
        pa = params["A"][mod]
        pb = state["B"][mod]
        pc = params["C"][mod]
        pd = params["D"][mod]
        pe = state["E"][mod]
        u = params["U"][mod]
        
        step_a,step_b,step_d = vectorize_weights(pa,pb,pd,u)
        step_c,step_e = to_log_space(pc,pe)
        step_a_nov,step_b_nov = get_vectorized_novelty(pa,pb,u,compute_a_novelty=False,compute_b_novelty=True)
            
            
            
        # Prior : even given by previous action or hard set if it is the first action
        d_dim_norm,_ = _normalize(step_d)
        no_context_prior = d_dim_norm
        with_context_prior = jnp.einsum("iju,j,u->i",step_b,previous_posterior_mod,last_action_mod)           
        prior_dim = no_context_prior*(1.0 - there_was_a_last_action_mod) + there_was_a_last_action_mod*with_context_prior
        
        
        # posterior_mod,F_mod = compute_state_posterior(prior_dim,[current_gauge_level],step_a)  # Given the available observation, I believe I am here !
        # state["previous_posterior"][mod] = posterior_mod
        
        # Active Inference Belief Update + action selection (?)
        end_of_trial_filter = jnp.ones((planning_options[mod]["horizon"]+2,))
        posterior_mod,F_mod,raw_qpi,efe_mod = compute_step_posteriors(t,prior_dim,current_stimuli,
                                    step_a,step_b,step_c,step_e,
                                    step_a_nov,step_b_nov,
                                    end_of_trial_filter,
                                    None,planning_options[mod])       
        efes[mod] = [efe_mod,step_e]

        # OPTIONAL : ONLINE UPDATING OF PARAMETERS 
        if model_options["learn_during_trials"] :  
            
            lr_b = step_parameters[mod]["transition_learning_rate"]
            fr_b = step_parameters[mod]["transition_forgetting_rate"]
            
            
            learn_what={"a":False,"b":True,"e":False}
            learn_rates={"a":0.0,"b":lr_b,"e":0.0}
            forget_rates = {"a":0.0,"b":fr_b,"e":0.0}
            if model_options["learn_habits"]:
                learn_what["e"] = True
                learn_rates["e"] = mod_parameters["habits_learning_rate"]
                forget_rates["e"] = 0
                
            
            new_pa,new_pb,new_pe = learn_during_trial([current_gauge_level],previous_posterior_mod,posterior_mod,last_action_mod,
                            pa,pb,pe,u,
                            learn_what=learn_what,
                            learn_rates=learn_rates,
                            forget_rates = forget_rates,
                            generalize_state_function=learning_options[mod]["state_generalize_function"])
            state["B"][mod] = new_pb
            state["E"][mod] = new_pe
        
        
        state["previous_posterior"][mod] = posterior_mod
        
        
        if not(model_options["modality_selector"] is None):
            # We need a way to compute an estimator of how much this modality
            # seems to control the gauge ! 
            if model_options["modality_selector"]["metric"] == "js_controll":
                
                _,new_step_b,_ = vectorize_weights(pa,pb,pd,u)
                
                d_omega[mod] = compute_js_controllability(new_step_b) 
                
            elif model_options["modality_selector"]["metric"] == "efe":
                action_distribution = jax.nn.softmax(mod_parameters["beta_pi"]*efe_mod)
                
                d_omega[mod] = jnp.inner(action_distribution,efe_mod) 
                    # Expected (neg) EFE for this modality
                    # The higher, the better !
                    # To be compared with the EFE from other modalities
                
            elif model_options["modality_selector"]["metric"] == "surprisal":
                d_omega[mod] = F_mod

            else :
                raise NotImplementedError("Unrecognized action modality selection metric : {}".format(model_options["modality_selector"]["metric"]))
    

    # If we use a modality selector, action selection is not independent across dimensions
    # but governed by an "omega" parameter. 
    reporting_data["efe"] = efes
    
    # Updating our appreciation of omega !
    if not(model_options["modality_selector"] is None):
        d_omega_vector = jnp.array(list(d_omega.values())) # Modality selector estimator for this step
        
        if model_options["modality_selector"]["learn"]: # Smooth update
            state["omega"] = omega + hyperparameters["alpha_omega"]*(d_omega_vector-omega)   # Update the omega vector to better match the expected controllability for each dimension
        else : # Point update
            # We only use our point estimate of omega at this specific timestep
            state["omega"] = d_omega_vector
            
        soft_omega_values = jax.nn.softmax(hyperparameters["beta_omega"]*state["omega"])
        softmaxed_omega = {key:omega_values for key,omega_values in zip(ACTION_MODALITIES,soft_omega_values)}         
    
    
    # Computing the actual action posterior : 
    final_action_distribution = {}
    for mod in ACTION_MODALITIES:
        mod_parameters = step_parameters[mod]

        # B/ PLANNING
        [efe_mod,habits_mod] = efes[mod]
        
        if not(model_options["modality_selector"] is None):
            attention_weight = softmaxed_omega[mod]
        else :
            attention_weight = 1.0
                
        action_distribution_mod = jax.nn.softmax(mod_parameters["beta_pi"]*(attention_weight*(efe_mod-habits_mod) + habits_mod))
                
        final_action_distribution[mod] = action_distribution_mod
        
    # The only thing missing from the state is the true action !
    return state,final_action_distribution,reporting_data



def actor_step(observation,state,params,rngkey,
                   hyperparameters,
                   planning_options,learning_options,
                   model_options):
    
    gauge_level,reward,t = observation
    # augmented_observation = (gauge_level,reward,None,t)
    
    new_state,action_posteriors,other_data = compute_action_posteriors(observation,state,params,hyperparameters,
                                                                       planning_options,learning_options,
                                                                       model_options)

    action_selected,vect_action_selected = sample_dict_of_categoricals(action_posteriors,rngkey)   
    
    # Udpate the last action with the sampled value :
    new_state["previous_action"] = vect_action_selected
    
    return new_state, (action_posteriors,action_selected,vect_action_selected), other_data


def predict(observation,state,params,
                   hyperparameters,
                   planning_options,learning_options,
                   model_options):
    
    gauge_level,obs_bool_filter,reward,true_action,t = observation  
    reduced_observation = (gauge_level,reward,t)
    
    new_state,action_posteriors,other_data = compute_action_posteriors(reduced_observation,state,params,
                                                                       hyperparameters,
                                                                       planning_options,learning_options,
                                                                       model_options)
    
    # Udpate the last action with the observed value :
    new_state["previous_action"] = true_action
    
    return new_state, action_posteriors, other_data
