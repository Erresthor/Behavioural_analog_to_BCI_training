
import sys,os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly as pltly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.mixture import GaussianMixture

from jax import vmap
import jax.numpy as jnp

# + local functions : 
from database_handling.database_extract import get_all_subject_data_from_internal_task_id
from utils import remove_by_indices
from analysis_tools.preprocess import get_preprocessed_data_from_df

def get_results_df(_internal_task_id,_studies_list = None,_exclude_subjects_list = [],
                   _llm_classification_code = None, _llm_classification_file_path = None,
                   _bins_fb_noise = None,_override = False,
                   last_k_trials= 2,last_t_timesteps=5):
    
    if _studies_list is not None :
        # Get a list of the task results, 
        _tasks_results_all = []
        for prolific_study_id in _studies_list:
            task_results = get_all_subject_data_from_internal_task_id(_internal_task_id,prolific_study_id,
                                                                    process_feedback_data_stream=True,override_save=_override)
            print(" - Loaded the task results for study {} \n    ({} subjects.)".format(prolific_study_id,len(task_results)))
            _tasks_results_all += task_results
    else :
        _tasks_results_all = get_all_subject_data_from_internal_task_id(_internal_task_id,
                                                                        process_feedback_data_stream=True,override_save=_override)
        
    print("Total : {} subjects".format(len(_tasks_results_all)))



    # Each subject in task results has the following entries : 
    # TASK_RESULT_FEATURES, TASK_RESULTS_EVENTS, TASK_RESULTS_DATA, TASK_RESULTS,RT_FB
    
    # let's remove some subjects based on broad inclusion criteria : 
    # did not do the task twice, did not revoke the consent midpoint, etc.
    remove_these_subjects = []
    for index,entry in enumerate(_tasks_results_all):
        subj_dict,_,_,_ = entry
        subj_name = subj_dict["subject_id"]
        if subj_name in _exclude_subjects_list:
            remove_these_subjects.append(index)

    _tasks_results_all = remove_by_indices(_tasks_results_all,remove_these_subjects)
    print(str(len(_tasks_results_all)) + " subjects remaining after removing problematic subjects.")
    
    
    
    # Fill a dataframe with that data :
    
    
    
    # The initial datframe is the first tuple in our task result list of tuples : 
    subjects_df = pd.DataFrame([entry[0] for entry in _tasks_results_all])

    # Avoid too many categories : 
    subjects_df['Sex'] = np.where(subjects_df['Sex'].isin(['Male','Female']), subjects_df['Sex'], 'Other')

    category_counts = subjects_df['Nationality'].value_counts()
    threshold = 2
    subjects_df['Nationality_red'] = subjects_df['Nationality'].apply(lambda x: x if category_counts[x] >= threshold else 'Other')

    # There was a single noise term for the whole training for each subject : 
    subject_noise_parameters = [np.array(entry[2]["parameters"]["noise_int"])[0] for entry in _tasks_results_all]

    # We add it to the df : 
    subjects_df["feedback_noise_std"] = subject_noise_parameters

    # Time taken to solve the task 
    # Add the time taken recorded by the application : (a better measure than the one provided by Prolific for some reason)
    subjects_df["application_measured_timetaken"] = (pd.to_datetime(subjects_df["finish_date"])-pd.to_datetime(subjects_df["start_date"])).dt.total_seconds()

    
    
    
    
    
    
    
    
    
    # In this dataframe, we're interested in sorting various kinds of data from the trials : 
    # 1/ Data from the instruction phase
    # Load LLM classifications for text responses if they are available !
    if _llm_classification_code is not None :
        classification_instructions = {}
        try : 
            with open(_llm_classification_file_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                
            for question_code,question_contents in loaded_dict.items():
                
                subject_classifs = question_contents["results"][_llm_classification_code]    

                
                subjects_df[question_code] = subject_classifs
                classification_instructions[question_code] = question_contents["prompt"]
                # print(subject_classifs)
            
        except : 
            print("Failed to load LLM classifications.")
    

    # 2/ Data from the feedback gauge :
    # Timestep values :
    all_subject_scores = [subjdata[2]["scoring"] for subjdata in _tasks_results_all]
    subjects_df["raw_feedback_values"] = [subj_scores["feedback"] for subj_scores in all_subject_scores]
    # Real time gauge values :
    subjects_df["realtime_values"] = [subjdata[3][1] for subjdata in _tasks_results_all] # Each element is a list of list os arrays (with varying shape)

    # 3/ Data from the hidden grid :
    # The grid for a specific trial: 
    trial_grids = [entry[2]["process"]["grids"] for entry in _tasks_results_all]
    subjects_df["grid_layout"] = trial_grids
    # Position value :
    subject_positions = [entry[2]["process"]["positions"] for entry in _tasks_results_all]
    subjects_df["subject_positions"] = subject_positions

    goal_positions = [np.array(entry[2]["parameters"]["goal_pos"])[:,0,:] for entry in _tasks_results_all]
    subjects_df["goal_position"] = goal_positions

    def euclidian_distance(position,goal):
        return jnp.linalg.norm(position-goal,2)
    gs = trial_grids[0][0].shape
    maximum_euclidian_dist = euclidian_distance(jnp.array(gs) - jnp.ones((2,)),jnp.zeros((2,)))
    all_euclidian_distances = vmap(vmap(vmap(euclidian_distance,in_axes=(0,None))))(jnp.array(subject_positions),jnp.array(goal_positions))/maximum_euclidian_dist
    subjects_df["norm_distance_to_goal"] = list(all_euclidian_distances)


    # 4/ Data from the realized actions :

    # Actions performed : this encompasses the points dropped
    # But may also include temporal elements such as :
    # - the time taken to perform an actions (first point / second point)
    # - when the action was performed with regard to the gauge
    canvas_size = _tasks_results_all[0][0]["canvas_size"] # Constant across all subjects + conditions
    all_actions_data = np.stack([subjdata[2]["blanket"]["actions"] for subjdata in _tasks_results_all]).astype(float)

    Nsubj,Ntrials,Nactions,Npoints,Nfeatures = all_actions_data.shape
    # print(all_actions_data)
    # Normalize the point data :
    all_actions_data[...,0] = all_actions_data[...,0]/canvas_size[0]
    all_actions_data[...,1] = 1.0 - all_actions_data[...,1]/canvas_size[1]


    # First, let's get a mask for all actions that were NOT performed :
    mask = all_actions_data[...,-1]==1  # values are 1 if the point was recorded
    both_points_only = (mask[...,0] & mask[...,1])
        # All points where at least one value is missing

    Nactions = all_actions_data[...,0,0].size
    Nmissed_actions = (~both_points_only).sum()
    print("A total of {}/{} actions were missed. ({:.2f} %)".format(Nmissed_actions,Nactions,100*Nmissed_actions/Nactions))

    subjects_df["raw_points"] = list(all_actions_data)


    # Encoded barycenters :
    barycenter_x = (all_actions_data[...,0,0]+all_actions_data[...,1,0])/2.0
    barycenter_y = (all_actions_data[...,0,1]+all_actions_data[...,1,1])/2.0
    barycenters = np.stack([barycenter_x,barycenter_y],axis=-1)
    subjects_df["action_barycenters"] = list(barycenters)

    # Encoded euclidian distance between points :
    action_distances = np.linalg.norm(all_actions_data[...,0,:2]-all_actions_data[...,1,:2],axis=-1)
    subjects_df["action_distances"] = list(action_distances)

    # Encoded evolution of point angles :
    angles = np.atan2(all_actions_data[...,1,1]-all_actions_data[...,0,1],all_actions_data[...,1,0]-all_actions_data[...,0,0])
    subjects_df["action_angles"] = list(angles)

    # Encoded delays between stimuli, point1 and point2 :
    all_action_delays = all_actions_data[...,-1,2]
    unfit_actions = (all_action_delays<10)
    subjects_df["action_time_between_points"] = np.where(all_action_delays>10, all_action_delays, np.nan).tolist()

    # Performance metric : we use the average distance to goal state across the lask k_T trials and the last k_t timesteps : (ignoring the blind trial)
    all_distances_to_goal_final = np.mean(np.stack(subjects_df["norm_distance_to_goal"])[:,-last_k_trials:-1,-last_t_timesteps:],axis=(-1,-2))
    subjects_df["final_performance"] = (1.0 - all_distances_to_goal_final).tolist()
    
    all_distances_to_goal_initial = np.mean(np.stack(subjects_df["norm_distance_to_goal"])[:,0,-last_t_timesteps:],axis=(-1))
    subjects_df["initial_performance"] = (1.0 - all_distances_to_goal_initial).tolist()

    # And for the blind trial :
    blind_trial_distances_to_goal = np.mean(np.stack(subjects_df["norm_distance_to_goal"])[:,-1,-last_t_timesteps:],axis=(-1))
    subjects_df["blind_trial_performance"] = (1.0 - blind_trial_distances_to_goal).tolist()
    
    
    
    
    
    # In our situation, the variables of interest are : 
    # a/ The level of noise of the gauge
    # b/ The performance of the subject
    # Let's define broad categories to classify them a bit easier :
    if _bins_fb_noise is None:
        subjects_df['noise_category'] = pd.cut(subjects_df['feedback_noise_std'], bins=[0,0.05,0.15,1.0], labels=["Low", "Medium", "High"])
    else : 
        subjects_df['noise_category'] = pd.cut(subjects_df['feedback_noise_std'], bins=_bins_fb_noise, labels=["Low", "Medium", "High"])
    
    _bins_performance_data = np.linspace(0,1,4)
    subjects_df['hard_performance_category'] = pd.cut(subjects_df['final_performance'], bins=_bins_performance_data, labels=["Poor", "Middling", "Good"])
    
    return subjects_df,classification_instructions
    
    
    


def get_full_dataframe_from_raw_data(studies_extraction_dictionnary,llm_classif_path,
                                     last_t_timesteps=5,last_K_trials=2,
                                     show_clustering = True):
    full_dataframe = pd.DataFrame()
    for study_name,study_codes in studies_extraction_dictionnary.items() :

        dataframe,qsts = get_results_df(study_codes["internal_task_id"],_studies_list = study_codes["studies_id"],_exclude_subjects_list = study_codes["exclude_subjects"],
                        _llm_classification_code = study_codes["dict_code"], _llm_classification_file_path = llm_classif_path,
                        _bins_fb_noise = study_codes["feedback_noise_bins"],last_t_timesteps=last_t_timesteps,last_k_trials=last_K_trials,_override = False)
        dataframe["study_name"] = study_name
        
        full_dataframe = pd.concat([full_dataframe,dataframe],ignore_index=True)




    # analyzed_dataframe = full_dataframe[(full_dataframe["study_name"] == "study_3" ) | (full_dataframe["study_name"] == "study_2")]  


    # For the performance, it's a bit more tricky : 
    # We can use the linear approximation or try to clusterize our results !
    override_linear = True


    # We use a simple clustering algorithm to define the categories :
    fit_this = np.expand_dims(full_dataframe["final_performance"].to_numpy(),-1)


    gm = GaussianMixture(n_components=3, random_state=0).fit(fit_this)
    # Print the initial parameters

    weights = gm.weights_
    means = gm.means_.flatten()
    sorted_indices = np.argsort(means)
    sorted_means = means[sorted_indices]
    covariances = gm.covariances_.flatten()

    # Create a new GaussianMixture with the same parameters but no fitting
    gmm_fixed = GaussianMixture(n_components=3, random_state=42)
    gmm_fixed.means_ = gm.means_[sorted_indices]
    gmm_fixed.covariances_ = gm.covariances_[sorted_indices]
    gmm_fixed.weights_ = gm.weights_[sorted_indices]
    gmm_fixed.precisions_cholesky_ = gm.precisions_cholesky_[sorted_indices] # Needed to avoid re-fitting

    labels = gmm_fixed.predict(fit_this)
    full_dataframe["label_idx"] = labels

    # Get the posterior probabilities for each data point (classification probabilities)
    posteriors = gmm_fixed.predict_proba(fit_this)
    print(posteriors.shape)
    full_dataframe["poor_performance_rating"] = list(posteriors[:,0])
    full_dataframe["middling_performance_rating"] = list(posteriors[:,1])
    full_dataframe["good_performance_rating"] = list(posteriors[:,2])
    def label_mapping(i) :
        if i == 0 :
            return "Poor"
        elif i == 1 :
            return "Middling"
        else :
            return "Good"
        
    full_dataframe["performance_category"] = full_dataframe["label_idx"].apply(label_mapping)

    if show_clustering : 
        sorted_means = means[sorted_indices]
        sorted_weights = weights[sorted_indices]
        sorted_covariances = covariances[sorted_indices]


        # Visualize the fitted mixture of gaussians :
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        density = np.exp(gm.score_samples(x))  # Convert log-likelihood to density
        def gaussian_pdf(x, mean, var):
            return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)
        individual_densities = [
            sorted_weights[i] * gaussian_pdf(x.flatten(), sorted_means[i], sorted_covariances[i])
            for i in range(gm.n_components)
        ]
        for i, (ind_density,cat_label) in enumerate(zip(individual_densities,["Poor performance","Middling performance","Good performance"])):
            plt.plot(x, ind_density, linestyle='--', linewidth=2, label=cat_label)
        # Histogram of the data
        plt.hist(fit_this, bins=30, density=True, alpha=0.6, color='gray', label='Histogram')
        # GMM density
        plt.plot(x, density, color='red', linewidth=2, label='GMM Density')
        plt.title('Visualization of GMM with Individual Gaussians')
        plt.xlabel('Final performance ($= 1 - DtG$) across the last {} timesteps'.format(last_t_timesteps))
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        all_vals = np.linspace(0,1.0,1000)
        posteriors = gmm_fixed.predict_proba(np.expand_dims(all_vals,-1))
        fig,ax = plt.subplots(1,1,figsize = ((6,6)))
        fig.suptitle("Clustering the subjects in 3 performance categories")
        ax.plot(all_vals,posteriors[:,2],color="green",label="good performers")
        ax.plot(all_vals,posteriors[:,1],color="orange",label="middling performers")
        ax.plot(all_vals,posteriors[:,0],color="blue",label="poor performers")
        sns.scatterplot(data=full_dataframe,x="final_performance",y="good_performance_rating",color="green")
        sns.scatterplot(data=full_dataframe,x="final_performance",y="middling_performance_rating",color="orange")
        sns.scatterplot(data=full_dataframe,x="final_performance",y="poor_performance_rating",color="blue")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel("Performance category posterior")
        plt.xlabel("Avg. performance across the last {} timesteps & {} trials".format(last_t_timesteps,last_K_trials))
        plt.show()
        
        
            
    
        # Effectives :   
        # Count the number of entries for each category
        category_counts = full_dataframe['performance_category'].value_counts().reset_index()
        category_counts.columns = ['performance_category', 'Count']# Create a barplot using seaborn
        sns.barplot(data=category_counts, x='performance_category', y='Count',hue ='performance_category',
                    order=["Poor","Middling","Good"],hue_order=["Poor","Middling","Good"],errorbar='sd')
        plt.ylabel("Subject Counts")
        plt.xlabel("Final performance category".format(last_t_timesteps,last_K_trials))
        plt.show()
    

    return full_dataframe