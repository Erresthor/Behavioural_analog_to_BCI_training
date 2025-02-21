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
from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.layer_trial import compute_step_posteriors
from actynf.jaxtynf.layer_learn import learn_after_trial
from actynf.jaxtynf.layer_options import get_learning_options,get_planning_options
from actynf.jaxtynf.shape_tools import to_log_space,get_vectorized_novelty
from actynf.jaxtynf.shape_tools import vectorize_weights

ACTION_MODALITIES = ["position","angle","distance"]

def update_params(trial_history,old_params,
                  learning_options,
                  model_options):
    # Learning, but in between trials ! 
    rewards,observations,states,actions = trial_history
    
    obs_vect_arr = [jnp.array(observations[0])]
    
    
    
    
    updated_params,reporting_data = {},{"smoothed_posteriors":{}}
    
    updated_params["A"] = old_params["A"]
    updated_params["C"] = old_params["C"]
    updated_params["U"] = old_params["U"]
    
    
    updated_b,updated_d,updated_e = {},{},{}
    for mod in ACTION_MODALITIES:
        qs_arr = jnp.stack(states["previous_posterior"][mod])
        u_vect_arr = jnp.stack(actions[mod])  
        
        pa = old_params["A"][mod]
        pb = old_params["B"][mod]
        pc = old_params["C"][mod]
        pd = old_params["D"][mod]
        pe = old_params["E"][mod]
        u = old_params["U"][mod] 
        
        learning_options_mod = learning_options[mod]
            
        # Then, we update the parameters of our HMM model at this level
        # We use the raw weights here !
        a_post,b_post,c_post,d_post,e_post,qs_post = learn_after_trial(obs_vect_arr,qs_arr,u_vect_arr,
                                                pa,pb,pc,pd,pe,u,
                                                method = learning_options_mod["method"],
                                                learn_what = learning_options_mod["bool"],
                                                learn_rates = learning_options_mod["learning_rates"],
                                                forget_rates= learning_options_mod["forgetting_rates"],
                                                generalize_state_function=learning_options_mod["state_generalize_function"],
                                                generalize_action_table=learning_options_mod["action_generalize_table"],
                                                cross_action_extrapolation_coeff=learning_options_mod["cross_action_extrapolation_coeff"],
                                                em_iter = learning_options_mod["em_iterations"])
        
       
        updated_b[mod] = b_post
        updated_d[mod] = d_post
        updated_e[mod] = e_post
        reporting_data["smoothed_posteriors"][mod] = qs_post
        
        
    updated_params["B"] = updated_b
    updated_params["D"] = updated_d
    updated_params["E"] = updated_e
      
    if not(model_options["modality_selector"] is None):        
        updated_params["omega"] = states["omega"][-1] 
            
            
    return updated_params,reporting_data
