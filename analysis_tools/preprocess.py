# Import the needed packages 
# 
# 1/ the usual suspects
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import jax.numpy as jnp

from .decompose_action import decompose_all_actions
from .decompose_feedback import decompose_all_observations

OPTIONS_PREPROCESS_DEFAULT = {
    "actions":{
        "distance_bins" : np.array([0.0,0.05,0.3,0.6,jnp.sqrt(2) + 1e-10]),
        "angle_N_bins"  : 4,
        "position_N_bins_per_dim" : 3
    },
    "observations":{
        "N_bins" : 5,
        "observation_ends_at_point" : 2
    }
}

def get_preprocessed_data(input_trial_data,input_feedback_series,
            options = OPTIONS_PREPROCESS_DEFAULT,
            verbose=True,
            autosave=True,autoload=True,override_save=False,label="default"):
    
    data_savefolder = os.path.join("ressources","preprocessed")
    data_savepath = os.path.join(data_savefolder,label+".data")
    
    if autoload:
        # Check if the file exists
        is_exists = os.path.isfile(data_savepath)
        
        # When can we just load ? When there is already a file and we do not override:
        just_load_and_return = (is_exists) and not(override_save)
        
        if just_load_and_return :
            # Just load and return the results then !
            with open(data_savepath, "rb") as input_file:
                data = pickle.load(input_file)
            return data
    
    # Do the actual preprocessing if there are not existing data or we are not allowed to use it (autoload false)
            
    # Actions : 
    ALL_ACTIONS = np.stack([subjdata["blanket"]["actions"] for subjdata in input_trial_data]).astype(float)
    indexes,vectorized_indexes,valid_actions = decompose_all_actions(ALL_ACTIONS,
                                    distance_bins=options["actions"]["distance_bins"],
                                    angle_N_bins=options["actions"]["angle_N_bins"],
                                    position_N_bins_per_dim=options["actions"]["position_N_bins_per_dim"])
    pos_index,dist_idx,angle_idx = indexes
    pos_vec,dist_vec,angle_vec = vectorized_indexes
    if verbose :
        Nactions = float(valid_actions.flatten().shape[0])
        Nvalid = np.sum(valid_actions)
        ratio = Nvalid/Nactions
        print("Out of the {} actions performed by our subjects, {} were 'valid' ({:.1f} %)".format(Nactions,Nvalid,100*ratio))
    
    # Observations :
    obsbool,obshist,obsscalar,obsseries = decompose_all_observations(input_feedback_series,input_trial_data,
                                                    N_FB_BINS=options["observations"]["N_bins"],
                                                    observation_ends_at_action=options["observations"]["observation_ends_at_point"])
    if verbose:
        Nobservations = float(obsbool.flatten().shape[0])
        Nvalid = np.sum(obsbool)
        ratio = Nvalid/Nactions
        print("Out of the {} feedback sequences potentially observed by our subjects, {} were 'valid' ({:.1f} %)".format(Nobservations,Nvalid,100*ratio))

    
    
    DATA = {
        "observations" : {
            "bool":obsbool,
            "vect":obshist,
            "idx" :obsscalar,
            "raw_series" :obsseries
        },
        "actions":{
            "bool":valid_actions,
            "idx":{
                "position":pos_index,
                "distance":dist_idx,
                "angle":angle_idx
            },
            "vect":{
                "position":pos_vec,
                "distance":dist_vec,
                "angle":angle_vec
            }
        }
    }
    
    if autosave:
        # If the directory does not exist, let's create it !
        os.makedirs(data_savefolder, exist_ok=True)
        
        with open(data_savepath, 'wb') as handle:
            pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DATA
    