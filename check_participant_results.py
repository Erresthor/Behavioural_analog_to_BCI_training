from plotting.summary.draw_timeline import draw_timeline
from plotting.summary.draw_performances import draw_perf
from plotting.summary.draw_actions import draw_actions

from database_handling.database_extract import extract_subject_data
from database_handling.access_remote_collections import get_complete_collection,get_partial_collection
""" 
A set of functions that should return a visible set of figures and a log to the experimenter, so that he may decide wether or not to reward the participant
"""
CANVAS_X,CANVAS_Y = 750,750
  
def check_incomplete_recordings_for_subject(client,subject_id):
    db_incomplete = client["master_inc"]
    collection_incomplete = db_incomplete["subjectdataincompletes"]
    
    matching_subjects_inc = collection_incomplete.find({"subjectId":subject_id})
    return [show_summary(subj_data) for subj_data in matching_subjects_inc]

def show_summary(data,incomplete=False,show_figures=True):
    assert data["bruh"]=="bruh_indeed","Key did not match. What is this doing here ? Are you sure this is a data from the good experiment ?" 
    
    
    dictionnary_instance,events,trials_data,fb_rtv = extract_subject_data(data,auto_translate= True)

    log_string ="""
    __________________________________________________________________
    subject | {subjname} | 
    task - {taskcode}
    language - {lang}
    browser - {brows}
    __________________________________________________________________
    STATUS : {completedTask}
    
    TIMING :
      - Started : {startdate}
      - Ended : {enddate}
    -> Total : {duration}
    
    WARNINGS : 
    > Exited fullscreen {n_exit_fs} times.
    > Missed {n_miss_action} actions.
    
    ANSWERS : 
    - Mandatory q1 : 
    {q11}
    - Mandatory q2 : 
    {q12}
    - Optional q1 : 
    {q21}
    - Optional q2 : 
    {q22}
    - Optional q3 : 
    {q23}
    
    """.format(subjname=dictionnary_instance["subject_id"],
               taskcode=dictionnary_instance["task_code"],
               lang=dictionnary_instance["lang"],
               brows=dictionnary_instance["browser"],
               completedTask=dictionnary_instance["completedTask"],
               startdate=dictionnary_instance["start_date"],
               enddate=dictionnary_instance["finish_date"],
               duration=dictionnary_instance["finish_date"]-dictionnary_instance["start_date"],
               n_exit_fs=dictionnary_instance["N_exited_fullscreen"],
               n_miss_action=dictionnary_instance["N_missed_actions"],
               q11=dictionnary_instance["mandatory_q1"],
               q12=dictionnary_instance["mandatory_q2"],
               q21=dictionnary_instance["optional_q1"],
               q22=dictionnary_instance["optional_q2"],
               q23=dictionnary_instance["optional_q3"]
               )
    
    if show_figures:
        final_clock = dictionnary_instance["finished_clock"]
        timeline_figure = draw_timeline(trials_data["timing"]["trial"]["start"],trials_data["timing"]["trial"]["end"],
                                        trials_data["timing"]["action"]["start"],trials_data["timing"]["action"]["end"],
                                        events,final_clock)
        timeline_figure.suptitle("Timeline : " + dictionnary_instance["subject_id"], fontsize=16)
        timeline_figure.show()
        
        try :
            feedback_levels_figure = draw_perf(trials_data["scoring"]["scores"],trials_data["scoring"]["feedback"],
                                            trials_data["timing"]["trial"]["start"],trials_data["timing"]["trial"]["end"],
                                            trials_data["timing"]["gauge_animation_end"],
                                            fb_rtv)
            feedback_levels_figure.suptitle("Performance : " + dictionnary_instance["subject_id"], fontsize=16)
            feedback_levels_figure.show()
        except:
            print("Could not plot feedback figure")
        
        try :
            actions_figure = draw_actions(trials_data["blanket"]["actions"],CANVAS_X,CANVAS_Y)
            actions_figure.suptitle("Actions : " + dictionnary_instance["subject_id"], fontsize=16)
            actions_figure.show()
        except:
            print("Could not plot actions figure")
    return log_string

def check_subject_recordings(subject_id,check_incomplete = False,all_incompletes = False):
    
    log_str = "Log for subject " + str(subject_id) + "\n\n"
    
    # In complete results : 
    collection_complete = get_complete_collection()
    
    matching_subjects = list(collection_complete.find({"subjectId":subject_id}))
    N_matching_subjs = len(matching_subjects)
    log_str +=("-------------------COMPLETE ENTRIES---------------------\n")
    log_str +=("Found {n_entries} entries matching for subject [{subj}]\n".format(n_entries=N_matching_subjs,subj=subject_id))
    
    if (N_matching_subjs==0) or (check_incomplete):
        log_str +=("\nChecking in the incomplete database as well ... \n")
        log_str +=("-------------------INCOMPLETE ENTRIES---------------------\n")
        
        # In incomplete results :
        collection_incomplete = get_partial_collection()
        
        matching_subjects_inc = list(collection_incomplete.find({"subjectId":subject_id}))
        N_matching_subjs_inc = len(matching_subjects_inc)
        
        log_str +=("Found {n_entries} INCOMPLETE entries matching for subject [{subj}]\n".format(n_entries=N_matching_subjs_inc,subj=subject_id))
        if N_matching_subjs_inc>0:
            # Show the first incomplete data found
            if all_incompletes:
                for subj_entry in matching_subjects_inc:
                    log_str+=show_summary(subj_entry,incomplete=True)
            else :
                log_str+=show_summary(matching_subjects_inc[-1],incomplete=True)
        if (N_matching_subjs==0):
            log_str +=("\n\nTrial not found for subject " + str(subject_id))
            log_str +=("\n---> Please review this case manually.")
            return log_str
        # raise ValueError("Trial not found for subject " + str(subject_id))
    
    if (N_matching_subjs>1):
        log_str +=("The following starting - ending dates were found:\n")
        for data in matching_subjects :
            log_str +=("  - " + str(data["firstTime"]) + " --- " + str(data["object_save_date"])+ "\n")
        log_str += ("\n\nMore than one entry with the same subject name (for subject " + str(subject_id) + ")")
        
        log_str += "\n"*3 + "LAST ENTRY : \n"
        # Get the full summary for the last subject : 
        log_str+=show_summary(matching_subjects[-1],incomplete=False)
        log_str += "\n"*3
        
        
        log_str +=("\n---> Please review this case manually.")
        return log_str
    
    # Get the full summary for the subject : 
    log_str+=show_summary(matching_subjects[-1],incomplete=False)
    
    return log_str
  


    
if __name__=="__main__":
    # Should store that somewhere else !
    # from database_handling.access_remote_collections import load_client_address
    # CLIENT = pymongo.MongoClient(load_client_address()) 
            
    # subj = "5c9cb670b472d0001295f377"  # This one started the task, failed 5 attention checks instantly, and then started again...
    
    # REVIEW THESE TOMORROW :
    # subj = "6595ae358923ce48b037a0dc"


    # subj ="66019988a3d63339927cc80f" 
    # subj = "5f0559d1bc410b85707c58a3"

    subj = "665e8784960919e34257696d"

            
    print(check_subject_recordings(subj,True,True))
    input()
    exit()