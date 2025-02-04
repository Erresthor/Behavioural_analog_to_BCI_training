import sys,os
import pymongo
import numpy as np
import csv

from .access_remote_collections import get_complete_collection
from .utils import get_values_in_interval


from langdetect import detect
from deep_translator import GoogleTranslator
import pickle

def get_feedback_recording(fb_data):
    # All datapoints of the gauge value (~30 / sec)
    fb_rtv = []
    for fbv in fb_data:
        fb_rtv.append([fbv["t"],fbv["value"],fbv["real_value"]])
    return np.array(fb_rtv)

def process_real_time_feedback(feedback_rt_array,
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
    
    
    N_FB_BINS = 7

    i = 3
    subject_i_fbs = RT_FBS[i]
    subject_i_trial_data = TRIAL_DATA[i]

    subject_i_noise_intensity = TRIAL_DATA[i]["parameters"]["noise_int"][0]

    subject_i_timings = subject_i_trial_data["timing"]
    subject_i_start_trials = subject_i_timings["trial"]["start"]
    subject_i_end_trials = subject_i_timings["trial"]["end"]
    subject_i_start_tsteps = subject_i_timings["timestep"]["start"]
    subject_i_end_tsteps = subject_i_timings["timestep"]["end"]
    subject_i_misses_tracker = subject_i_timings["missed_actions"]

    subject_i_action_1_tstamps = subject_i_timings["action"]["start"]
    subject_i_action_2_tstamps = subject_i_timings["action"]["end"]

    feedback_series=  get_all_feedback_series(subject_i_fbs,
                                            subject_i_start_trials,subject_i_end_trials,
                                            subject_i_start_tsteps,subject_i_end_tsteps,
                                            subject_i_action_1_tstamps,subject_i_action_2_tstamps,
                                            subject_i_misses_tracker,
                                            observation_ends_at_action=2)





def open_timestep_data(list_of_tmstp,max_n_timesteps,
                       last_tick_obs,last_tick_pos,
                       paddit = False):
    """ Return array-like containers of timestep data. """
    
    # Things that can't (directly) be put into arrays : 
    total_timesteps = len(list_of_tmstp)
    grid_change = [tmstp_data["infered_grid_movement"] for tmstp_data in list_of_tmstp]  # Not really needed ?
    timestep_idxs = [tmstp_data["step"] for tmstp_data in list_of_tmstp]
    
    
    # Prompts : this should probably be done by another function
    prompted = [(tmstp_data["prompted"]!=None) for tmstp_data in list_of_tmstp] 
    #  (See this later) - Contents of a prompt when it is not "None" : 
    # prompt_showtime : Number,
    # prompt_clicktime : Number,
    # true_val : String,
    # subj_pred : String,
    # prompt_correct :Boolean
    
    # Timing elements :
    tmstps_starts = [tmstp_data["time_start"] for tmstp_data in list_of_tmstp]
    tmstps_ends = [tmstp_data["time_end"] for tmstp_data in list_of_tmstp]
    action_start = [tmstp_data["action_start"] for tmstp_data in list_of_tmstp]
    action_end = [tmstp_data["action_end"] for tmstp_data in list_of_tmstp]
    gauge_anim_end = [tmstp_data["gauge_animation_end"] for tmstp_data in list_of_tmstp]
    
    # Observations & actions
    true_feedback_values = [tmstp_data["feedbackValue"] for tmstp_data in list_of_tmstp]
    action_points = [tmstp_data["pointsData"] for tmstp_data in list_of_tmstp]
    grid_pos = [tmstp_data["grid_position"] for tmstp_data in list_of_tmstp]
    
    # Check if any action was missed :
    actions_missed = []
    for idx,pts in enumerate(action_points) :
        if (pts==[]):
            actions_missed.append(1)
            action_points[idx] = [[0,0,0,0],[0,0,0,0]]
        elif (len(pts)==1):
            actions_missed.append(0.5) # Missing 1 point ! 
            action_points[idx].append([0,0,0,0])
        else : 
            actions_missed.append(0)
            
    # Padding time ! If the trial ended early (most likely due to a successful result)
    # we can pad the results so that they may fit the table
    # Note : we should make some kind of filter afterwards so that it does not affect the fitting procedure
    if paddit :
        for k in range(total_timesteps,max_n_timesteps):
            tmstps_starts.append(tmstps_ends[-1])
            tmstps_ends.append(tmstps_ends[-1])
            action_start.append(tmstps_ends[-1])
            action_end.append(tmstps_ends[-1])
            gauge_anim_end.append(tmstps_ends[-1])
            
            true_feedback_values.append(last_tick_obs)
            action_points.append([[0,0,0,0],[0,0,0,0]])
                # Each point recorded has the following coordinates :
                # canvasMousePos.x,canvasMousePos.y,Date.now()-dboardloc.start_draw_time,1
            actions_missed.append(0)
            
            grid_pos.append(last_tick_pos)
            grid_change.append("same")
        
        true_feedback_values.append(last_tick_obs) #
        grid_pos.append(last_tick_pos)
    
    qualitatives = (grid_change,prompted)
    quantitatives = (tmstps_starts,tmstps_ends,action_start,action_end,gauge_anim_end,grid_pos,true_feedback_values,action_points,actions_missed)
    return quantitatives,qualitatives

def open_trial_parameters(list_of_trials):
    
    parameters = [trial_data["trialParameters"] for trial_data in list_of_trials]
    # print(parameters)
    # should be done for each trialParameters key : 
    # - start_pos (2D)
    # - end_pos   (2D) 
    # - gauge_freq  (1D)
    # - gauge_int   (1D)
    # - gauge_update_interval  (1D)
    # - smooth_edges (1D boolean)
    
    # For now, let's extract these ones : 
    gauge_int_all_trials = [trial_params["gauge_int"] for trial_params in parameters]
    goal_position_all_trials = [trial_params["end_pos"] for trial_params in parameters]
    
    return {
        "noise_int":gauge_int_all_trials,
        "goal_pos":goal_position_all_trials
    }
    
def open_trial_data(list_of_trials,n_timesteps_max,pad_timesteps=True):
    # Environment elements :
    trial_idxs = [trial_data["trialNumber"] for trial_data in list_of_trials]
    grids = np.array([trial_data["grid"] for trial_data in list_of_trials])    
    
    # Timing elements :
    trials_starts = np.array([trial_data["start_trial_time"] for trial_data in list_of_trials])
    trials_ends = np.array([trial_data["end_trial_time"] for trial_data in list_of_trials])
    
    
    # Timestep decomposition : (this is a bit more involved, as we need to account for
    # trials that ended early, in effect padding their values)
    all_feedback_values = []
    all_pos_values = []
    all_action_values = []
    
    all_t_tmstps_start = []
    all_t_tmstps_end = []
    all_t_action_starts = []
    all_t_action_end = []
    all_t_gauge_anim_end = []
    
    all_action_misses = []
    for trial in list_of_trials:
        quant,qual = open_timestep_data(trial["timesteps"],n_timesteps_max,
                                        trial["last_tick_obs"],trial["last_tick_pos"],pad_timesteps)
        
        (tmstps_starts,tmstps_ends,action_start,action_end,gauge_anim_end,
                                  grid_pos,true_feedback_values,action_points,actions_missed) = quant
        
        # We're especially interested in the obervables :
        # Concatenate the last observation and the last grid position (that is not a proper timestep) :
        # true_feedback_values.append(trial["last_tick_obs"])
        all_feedback_values.append(true_feedback_values)
        
        # grid_pos.append(trial["last_tick_pos"])
        all_pos_values.append(grid_pos)
        
        all_action_values.append(action_points)
        
        all_t_tmstps_start.append(tmstps_starts)
        all_t_tmstps_end.append(tmstps_ends)
        all_t_action_starts.append(action_start)
        all_t_action_end.append(action_end)
        all_t_gauge_anim_end.append(gauge_anim_end)
        
        all_action_misses.append(actions_missed)
        
        
    # Vectorize the quantities of interest ! (won't work without padding)
    if pad_timesteps:
        all_feedback_values = np.array(all_feedback_values)
        all_pos_values = np.array(all_pos_values)
        all_action_values = np.array(all_action_values)
        all_t_tmstps_start = np.array(all_t_tmstps_start)
        all_t_tmstps_end = np.array(all_t_tmstps_end)
        all_t_action_starts = np.array(all_t_action_starts)
        all_t_action_end = np.array(all_t_action_end)
        all_t_gauge_anim_end = np.array(all_t_gauge_anim_end)
        all_action_misses = np.array(all_action_misses)
    
    # Trial Performance indicators
    scores = np.array([trial_data["finalScore"] for trial_data in list_of_trials])
    
    # Can't be vectorized (directly): 
    success = [(trial_data["outcome"]=="success") for trial_data in list_of_trials]    
    
    
    timing = {
        "trial":{
            "start":trials_starts,
            "end": trials_ends
        },
        "timestep":{
            "start":all_t_tmstps_start,
            "end" :all_t_tmstps_end
        },
        "action":{
            "start" :  all_t_action_starts,
            "end": all_t_action_end
        },
        "gauge_animation_end":all_t_gauge_anim_end,
        "missed_actions" : all_action_misses
    }
    
    scoring = {
        "scores" : scores,
        "success" : success,
        "feedback" : all_feedback_values
    }
    
    process =  {
        "grids" : grids,
        "positions" : all_pos_values
    }
    
    blanket = {
        "feedback" : all_feedback_values,
        "actions" : all_action_values
    }
    
    trials_data = {
        "timing" : timing,
        "scoring" : scoring,
        "process" : process,
        "blanket" : blanket,
        "parameters":open_trial_parameters(list_of_trials)
    }

    return trials_data

def fetch_participant_info(prolific_exports_foldername,prolific_study_id,participant_prolific_id,ignore_if_did_not_find=True):
    export_name = "prolific_export_" + prolific_study_id + ".csv"
    filepath = os.path.join(prolific_exports_foldername,export_name)
    
    if not(os.path.isfile(filepath)):
        if ignore_if_did_not_find : 
            print("ALERT : Could not find the prolific export requested for task {}. Are you sure it was deposited in the folder {} ?".format(prolific_study_id,prolific_exports_foldername))
            return [],[]
        else :
            raise FileNotFoundError("Could not find the prolific export requested. Are you sure it was deposited in the folder " + prolific_exports_foldername + " ?" )
    
    with open(filepath, 'r') as file:
        csvreader = csv.reader(file)
        labels = next(csvreader)
        
        for row in csvreader:
            if participant_prolific_id in row:
                formatted_row = row
                for k,x in enumerate(formatted_row) :
                    try :
                        formatted_row[k] = float(x)
                    except:
                        continue
                return labels,formatted_row
    
    if ignore_if_did_not_find : 
        print("Could not find the prolific id for the following subject : " + participant_prolific_id + ". Are you sure the prolific export is up to date ?")
        return [],[]
    
    # No entry found    
    raise ValueError("Could not find the prolific id for the following subject : " + participant_prolific_id + ". Are you sure the prolific export is up to date ?" )

def extract_subject_data(data,auto_translate= True):
    assert data["bruh"]=="bruh_indeed","Key did not match. What is this doing here ? Are you sure this is a data from the good experiment ?" 
    
    # # If we are exploring the complete database,
    # # Basic checks : all of these should match : 
    task_did_not_end_early = (data["early_save_time"]==None)
    all_trials_recorded = (data["maxTRIALS"]==len(data["trialData"]))
    auto_check_all_went_well = task_did_not_end_early and all_trials_recorded
    
    # Additional data if needed :         
    start_clock = data["expe_start_time"]  # Absolute clock,not really relevant
    end_clock_complete = data["expe_end_time"]      # Relative clock
    end_clock_partial = data["early_save_time"]
    
    end_clock = end_clock_complete
    if end_clock==None:
        end_clock = end_clock_partial    
    
    events = data["events"] # Used for plotting,among others

    fullscreen_exit_events = [ev for ev in events['fullscreen'] if ev["val"]=="exited_fullscreen"]
    N_exit_fullscreen = len(fullscreen_exit_events)
    
    missed_action_events = [ev for ev in events['timesteps']]
    N_missed_actions = len(missed_action_events)
    
    
    # Auto translate questions fields     
    def transl(original_str):
        if original_str==None:
            original_str = "Did not answer."
        if auto_translate:
            language = detect(original_str)
            if language != "en":
                translation = GoogleTranslator(source='auto', target='en').translate(original_str) 
                return  translation + "  [TRANSLATED FROM ORIGINAL : " + original_str + " ]"
        return original_str
    
    # One field of the DB used to write the summary for any participant : 
    dictionnary_instance = {
        "subject_id" : data["subjectId"],
        "task_code" : data["taskCode"],
        "lang" : data["languageId"],
        "browser" : data["detected_browser"],
        
        "N_trials":data["maxTRIALS"],
        "N_tmstps":data["maxTMSTPS"],
        "start_date" : data["firstTime"],
        "finish_date" : data["object_save_date"],
        "finished_clock" : end_clock,
        
        "completedTask" : auto_check_all_went_well,
        "N_exited_fullscreen" : N_exit_fullscreen,
        "N_missed_actions" : N_missed_actions,  
        "canvas_explore_points" : data["explore_canvas_points"],
        
        "mandatory_q1" : data["feedback_control_est_question"] ,
        "mandatory_q2" : transl(data["feedback_control_text_question"]),
        "optional_q1" : transl(data["design_q1"]),
        "optional_q2" : transl(data["design_q2"]),
        "optional_q3" : transl(data["design_q3"]),
        
        "canvas_size" : (750,750) # Hard set values
    }
    
    # Trial data :
    trials_data = open_trial_data(data["trialData"],data["maxTMSTPS"],pad_timesteps=True)
    
    # Feedback gauuge data :
    # All datapoints of the gauge value (~30 / sec)
    fb_rtv = get_feedback_recording(data["feedbackRTValues"])
        # This will have to be integrated with the remainder of the data
    

    
    
    
    return dictionnary_instance,events,trials_data,fb_rtv

def get_full_subject_entry(recordings_collection,subject_id,task_id=None,
                           process_feedback_data_stream=False):    
    
    if task_id is None:
        matching_subjects = list(recordings_collection.find({"subjectId":subject_id}))
    else :
        matching_subjects = list(recordings_collection.find({"$and": [{"subjectId":subject_id}, {"taskCode":{"$regex": task_id}}]}))
    
    if len(matching_subjects)>1:
        for subj in matching_subjects:
            print(extract_subject_data(subj))
        print(matching_subjects)
        raise ValueError("More than one matching entry for subject " + str(subject_id))
        
    
    dictionnary_instance,events,trials_data,fb_rtv = extract_subject_data(matching_subjects[0])
    task_participated_in = dictionnary_instance["task_code"]

    [internal_task_reference,prolific_study_id] = task_participated_in.split("+")

    # 2. Get the data from prolific export database
    prolific_exports_foldername = os.path.join("ressources","prolific_exports")
    keys,vals = fetch_participant_info(prolific_exports_foldername,prolific_study_id,subject_id)
    subject_dict = dict(zip(keys,vals))  # To dictionnary
    
    # 3. Concatenate the data
    full_dict = dict(dictionnary_instance)
    full_dict.update(subject_dict)
    
    
    # 4. If demanded, cut the feedback datastream into trials and timesteps and store them with the rt_fb : 
    if process_feedback_data_stream:
        subject_i_timings = trials_data["timing"]
        subject_i_start_trials = subject_i_timings["trial"]["start"]
        subject_i_end_trials = subject_i_timings["trial"]["end"]
        subject_i_start_tsteps = subject_i_timings["timestep"]["start"]
        subject_i_end_tsteps = subject_i_timings["timestep"]["end"]
        subject_i_misses_tracker = subject_i_timings["missed_actions"]

        subject_i_action_1_tstamps = subject_i_timings["action"]["start"]
        subject_i_action_2_tstamps = subject_i_timings["action"]["end"]
        
        
        
        cut_fb_stream = process_real_time_feedback(fb_rtv,
                                subject_i_start_trials,subject_i_end_trials,
                                subject_i_start_tsteps,subject_i_end_tsteps,
                                subject_i_action_1_tstamps,subject_i_action_2_tstamps,
                                subject_i_misses_tracker,
                                observation_ends_at_action=2)
        
        
        
        fb_rtv = (fb_rtv,cut_fb_stream)
    
    return full_dict,events,trials_data,fb_rtv



def get_all_subject_data_from_internal_task_id(internal_task_id,prolific_task_id=None,
                    process_feedback_data_stream = False,           
                    autosave=True,override_save=False,autoload=True):
    check_prolific_task_id = not(prolific_task_id is None)
    
    filename = str(internal_task_id)
    
    if check_prolific_task_id :
        filename = filename + '_' + prolific_task_id
        
    data_savefolder = os.path.join("ressources","extracted")
    data_savepath = os.path.join(data_savefolder,filename+".data")
    
    
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
    
    
    # 1. Get the completed recordings from Atlas MongoDB :
    # (this operation may take a while for bad  connections or if we have a big amount of subjs)
    collection_complete = get_complete_collection()
        # All the data from the subjects we're interested in !

    # Ugly querying incoming : 
    # Go through the whole database (should not be THAT long, this is a pretty small DB)
    subject_ids_concerned = []
    for entry in collection_complete.find():
        # Find all the subjects with matching internal task id code ! 
        [recorded_task, recorded_prolific_task_id] = (entry["taskCode"].split("+"))
       
        if recorded_task == internal_task_id:
            if not(check_prolific_task_id) or (recorded_prolific_task_id==prolific_task_id):
                subject_ids_concerned.append(entry["subjectId"])
    
    return_data = []
    for subjid in subject_ids_concerned:
        return_data.append(get_full_subject_entry(collection_complete,subjid,internal_task_id,
                                                  process_feedback_data_stream))
    
    
    
    if autosave:
        # If the directory does not exist, let's create it !
        os.makedirs(data_savefolder, exist_ok=True)
        
        with open(data_savepath, 'wb') as handle:
            pickle.dump(return_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return return_data


if __name__=="__main__":   
            
    internal_task_id = '001'
    r = get_all_subject_data_from_internal_task_id(internal_task_id)
    print(r)
    exit()