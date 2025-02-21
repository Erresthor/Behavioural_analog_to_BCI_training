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


ACTION_MODALITIES = ["position","angle","distance"]

def update_params(trial_history,old_params,
                  model_options):
    # Learning, but in between trials ! 
    rewards,observations,states,actions = trial_history
    
    updated_params = {}
    if model_options["model_family"] == "random":
        return updated_params

    q_table = {}
    for mod in ACTION_MODALITIES:
        q_table[mod] = states["q_table"][mod][-1]
    updated_params["q_table"] = q_table
    
    if not(model_options["modality_selector"] is None):        
        updated_params["omega"] = states["omega"][-1] 
    
    if model_options["_track_transitions"] :
        updated_params["B"] = {}
        for mod in ACTION_MODALITIES:
            updated_params["B"][mod] = states["B"][mod][-1]
    
    if model_options["model_family"] == "latql" :
        updated_params["A"] = old_params["A"]
        updated_params["D"] = old_params["D"]
    
    return updated_params
