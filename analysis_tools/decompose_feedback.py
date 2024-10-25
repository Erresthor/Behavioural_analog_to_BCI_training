
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap

from functools import partial



# Utils, a priori only fit for these functions :
def get_values_in_interval(sorted_array,lower_bound=None,upper_bound=None,axis_searched=0):
    sort_axis = sorted_array[:,axis_searched]
    
    if lower_bound == None:
        lower_bound = np.min(sort_axis)
    if upper_bound == None:
        upper_bound = np.max(sort_axis)
    
    condition = (sort_axis>=lower_bound) & (sort_axis<=upper_bound)

    return sorted_array[condition]

def get_all_feedback_series(feedback_rt_array,
                            raw_trial_starts,raw_trial_ends,
                            raw_tmstp_starts,raw_tmstp_ends,
                            t_action_1,t_action_2,
                            misses_tracker,
                            observation_ends_at_action = 2,
                            INITIAL_TRIAL_START_OFFSET=2000):
    
    N_trials_visible = raw_trial_starts.shape[0]-1 # last trial was blind !
    Ntimesteps = raw_tmstp_starts.shape[-1] 
    
    
    # Only keep the feedbacks recorded after the start of the trial : (and the initial timesteps within 2000 ms !)
    task_start_t = raw_trial_starts[0] - INITIAL_TRIAL_START_OFFSET  # We change this value to use it when accounting for observed feedback values
    
    
    
    all_trial_starts = raw_trial_starts
    all_trial_starts[0] = task_start_t
    
    all_trial_ends = raw_trial_ends

    # Remove the feedback values seen during the instructions :
    task_fbs = get_values_in_interval(feedback_rt_array,task_start_t)
        

    feedback_series = []
    
    for trial_k in range(N_trials_visible):
        trial_start_t,trial_end_t = all_trial_starts[trial_k],all_trial_ends[trial_k]

        # Remove all data before this trial and all data after
        trial_feedbacks = get_values_in_interval(task_fbs,trial_start_t,trial_end_t)   
        
        # Let's try to keep only the arrays we're interested in (we could probably do this with a dataframe...)
        # The initial observation is everything between the start of the trial and the end of the first action
        observation_start_t,observation_end_t = trial_start_t,t_action_2[trial_k,0] # raw_tmstp_ends[trial_k,0]
        all_observation_arrays = [get_values_in_interval(trial_feedbacks,observation_start_t,observation_end_t)]
            
        
        
        
        for timestep_k in range(1,Ntimesteps):
            if (observation_ends_at_action == 2):
                # Observations can either be cut off after the second point :
                observation_start_t,observation_end_t = t_action_2[trial_k,timestep_k-1],t_action_2[trial_k,timestep_k]
            elif (observation_ends_at_action == 1) :
                # Or the first, depending on our model : 
                # (we consider the subject did not take into account gauge movements during the action)
                observation_start_t,observation_end_t = t_action_2[trial_k,timestep_k-1],t_action_1[trial_k,timestep_k]
            else : 
                raise NotImplementedError("Observation ends key error : only 1 or 2 are accepted.")
            
            if misses_tracker[trial_k,timestep_k]>0.51:
                # If we missed the action, the observation_end_t can be considered as the timestep end :
                observation_end_t =  raw_tmstp_ends[trial_k,timestep_k]

            all_observation_arrays.append(get_values_in_interval(trial_feedbacks,observation_start_t,observation_end_t))    
        
        
        # The very last observation is between the last action and the end of the trial !
        last_obs_start_t = t_action_2[trial_k,-1]
        last_obs_end_t = trial_end_t
        all_observation_arrays.append(get_values_in_interval(trial_feedbacks,last_obs_start_t,last_obs_end_t))   
        
        
        feedback_series.append(all_observation_arrays)
    
    # feedback_series is a [[np.ndarray]] object
    return feedback_series


            
def get_feedback_differences(feedback_series,true_feedback_values):
    # true_feedback_values = (subject_i_trial_data["scoring"]["feedback"][trial_k])
    Ntrials = len(feedback_series)
    
    
    difference_feedback_series = []
    
    for trial_k in range(Ntrials):
        trial_feedback_array = feedback_series[trial_k]
        trial_feedback_true = true_feedback_values[trial_k]
        
        Ntimesteps = len(trial_feedback_array) 
        print(Ntimesteps,trial_feedback_true)
        assert Ntimesteps == trial_feedback_true.shape[0],"Issue when unpacking true feedback value"


        difference_feedback_trial = []
        for tmstp_k in range(Ntimesteps):
            tmtsp_feedback_array = trial_feedback_array[tmstp_k]
            tmtsp_feedback_true_value = trial_feedback_true[tmstp_k]
            
            
            tmtsp_feedback_array_is_empty = tmtsp_feedback_array.shape[0]==0
            
            
            if not(tmtsp_feedback_array_is_empty):
                print(tmtsp_feedback_array)
                print(tmtsp_feedback_array.shape)
                # Center all observations on the true feedback value at this timestep
                norm_fb = tmtsp_feedback_array[:,1] - tmtsp_feedback_true_value
                # Center all observations on the moment the action was taken
                norm_time = tmtsp_feedback_array[:,0] - np.max(tmtsp_feedback_array[:,0])
                
                difference_feedback_trial.append(np.stack([norm_time,norm_fb],axis=-1))
            else :
                difference_feedback_trial.append(None)
        difference_feedback_series.append(difference_feedback_trial)
    
    return difference_feedback_series       
      
def decompose_real_time_feedback(_series,Nbins = 5,
                                 eps =1e-6,
                                 normalize_histogram = True): 
    bins = np.linspace(0.0,1.0+eps,Nbins+1)
        
    # Let's note which observations were not seen by the subjects : (0-shaped arrays)
    ope = (lambda x : 0 if x.shape[0]==0 else 1)
    seen_fb_raw = np.array([[ope(j) for j in i]  for i in _series])
    
    # We can assume that the first 'unseen' feedback after a trial was actually seen and interpreted as a 1.0 !
    y = np.roll(seen_fb_raw,1,axis=1)
    y[:,0] = 0
    seen_fb = np.where(seen_fb_raw+y>=1,1,0)

    # 1st : we can just pick the mean of each existing fb series, and digitize that :
    # This is very basic, and struggles to take into account : 
    # - possible noises in subject perception 
    # - temporal perceptive dynamics
    # - the movement of the feedback
    
    def _mean_if_exists(__arr):
        _mean_val = np.mean(__arr[:,1])
        return _mean_val
    scalar_means = np.array([[_mean_if_exists(j) for j in i]  for i in _series])
    # If we have nothing here, we may assume that the trial was successful :)
    scalar_means = np.where(np.isnan(scalar_means),1.0,scalar_means)
    
    
    # To make it fit the observation modality, we will need to digitize it and make it one_hot !
    dig_means = np.digitize(scalar_means,bins)-1
        
    # 2nd : digitize each timepoint individually, and then add all the digitized vectors together 
    # (possibly weighted to favor the most recent values ?)
    def _weighted_approach(__arr):
        _digitized_points = np.digitize(__arr[...,1],bins)-1
        
        # Build a histogram from each of those digitized values : 
        # We can even play with weights to account for more complex perception rules
        # (for now, no weights)
        obs_histogram = np.bincount(_digitized_points,minlength=Nbins)
        
        if(np.sum(obs_histogram)==0): # No points here ! Assume that we got the best possible observation
            obs_histogram[-1] = 1
        
        return obs_histogram
        
    points_histograms = np.array([[_weighted_approach(j) for j in i]  for i in _series])

    if normalize_histogram:
        colsum = points_histograms.sum(axis=-1)
        points_histograms = points_histograms / colsum[..., np.newaxis]
    return seen_fb,points_histograms,(scalar_means,dig_means)

def decompose_all_observations(feedbacks_series_all_subj,trial_datas_all_subj,N_FB_BINS=5,observation_ends_at_action=2):
    assert len(feedbacks_series_all_subj)==len(trial_datas_all_subj), "Input list length mismatch :("
    
    
    obs_bool_all,obs_histogram_all,obs_series_all,obs_scalar_all = [],[],[],[]
    for i,(subj_trial_datas,subj_feedback_series) in enumerate(zip(trial_datas_all_subj,feedbacks_series_all_subj)):
        subject_i_noise_intensity = subj_trial_datas["parameters"]["noise_int"][0]

        subject_i_timings = subj_trial_datas["timing"]
        subject_i_start_trials = subject_i_timings["trial"]["start"]
        subject_i_end_trials = subject_i_timings["trial"]["end"]
        subject_i_start_tsteps = subject_i_timings["timestep"]["start"]
        subject_i_end_tsteps = subject_i_timings["timestep"]["end"]
        subject_i_misses_tracker = subject_i_timings["missed_actions"]

        subject_i_action_1_tstamps = subject_i_timings["action"]["start"]
        subject_i_action_2_tstamps = subject_i_timings["action"]["end"]

        feedback_series=  get_all_feedback_series(subj_feedback_series,
                                                subject_i_start_trials,subject_i_end_trials,
                                                subject_i_start_tsteps,subject_i_end_tsteps,
                                                subject_i_action_1_tstamps,subject_i_action_2_tstamps,
                                                subject_i_misses_tracker,
                                                observation_ends_at_action=observation_ends_at_action)
        obs_bool,obs_histogram,obs_scalar = decompose_real_time_feedback(feedback_series,Nbins=N_FB_BINS)

        
        # Pack it :
        obs_bool_all.append(obs_bool)
        obs_histogram_all.append(obs_histogram)
        obs_series_all.append(feedback_series)
        obs_scalar_all.append(obs_scalar)
    return np.array(obs_bool_all),np.array(obs_histogram_all),np.array(obs_scalar_all),obs_series_all

if __name__ == '__main__':
    # We're also interested in the trial timing data, to know precisely when an action was conducted, and when the gauge started moving :
    

    i = 1     # 42 is 0, 0 is 0.23, 5 is 0.4

    subject_i_features = subjects_df.iloc[i]
    print("Noise intensity : {:.2f}".format(subject_i_features["feedback_noise_intensity"]))
    
