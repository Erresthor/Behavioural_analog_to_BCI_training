import sys,os
import pymongo
import numpy as np
import csv
import pickle 
import json

from database_handling.access_remote_collections import get_complete_collection

LOCAL_SAVE_PATH = os.path.join("ressources","local_experiment_saves","local_database.json") 

def save_full_collection_locally(filepath = LOCAL_SAVE_PATH):
    if os.path.exists(filepath):
        print(f"File '{filepath}' already exists.")  # Or prompt for action
        return
      
    # 1. Get the completed recordings from Atlas MongoDB :
    collection_complete = get_complete_collection()
    
    data = list(collection_complete.find())
    # Save the data to a JSON file
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, default=str)  # default=str handles ObjectId serialization

    print(f"Exported {len(data)} documents to 'local_collection.json'")
    return

def load_local_data_into_pymongo_collection(filepath = LOCAL_SAVE_PATH):
    
    with open(filepath, "r") as json_file:
        loaded_data = json.load(json_file)
    
    return loaded_data

# Here, we save the raw database (in case something bad happens to the remote database)
def save_collection_locally(internal_task_id,savepath=os.path.join("ressources","local_experiment_saves")):
    # 1. Get the completed recordings from Atlas MongoDB :
    collection_complete = get_complete_collection()
        # All the data from the subjects we're interested in !

    # Ugly querying incoming : 
    # Go through the whole database (should not be THAT long, this is a pretty small one)
    subject_ids_concerned = []
    for entry in collection_complete.find():
        # Find all the subjects with matching internal task id code ! 
        recorded_task = (entry["taskCode"].split("+"))[0]
        if recorded_task == internal_task_id:
            subject_ids_concerned.append(entry["subjectId"])
    
    if len(subject_ids_concerned)==0:
        print("No subjects found for the task" + internal_task_id + ", aborting !")
        return None,None
    
    save_this_locally = []
    for subj_id in subject_ids_concerned:
        matching_subjects = list(collection_complete.find({"subjectId":subj_id}))
        assert len(matching_subjects)==1,"More than one matching entry for subject " + str(subj_id)
        save_this_locally.append(matching_subjects[0])
    
    filepath = os.path.join(savepath,internal_task_id + '.pickle')
    
    is_exists = os.path.isfile(filepath)
    if (is_exists):
        accepted_deletion = False
        while(not(accepted_deletion)):
            input_value = input("Collection " + internal_task_id + " already found ! Override ? (y/n)")
            if input_value=="n":
                print("Aborting local collection save.")
                return None,None
            elif input_value=="y":
                accepted_deletion = True
            else:
                print("Please answer correctly...")
    
    with open(filepath, 'wb') as handle:
        pickle.dump(save_this_locally, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return save_this_locally,subject_ids_concerned



from database_handling.database_extract import get_all_subject_data_from_internal_task_id

if __name__ == "__main__":
    
    
    #             # Relative to the root folder
    
    # _,c = save_collection_locally("003",LOCAL_SAVE_PATH)
    # print(c)
    # print("("+str(len(c))+" subjects)")
    
        
    full_coll_from_local = get_all_subject_data_from_internal_task_id("002")
    
    print(len(full_coll_from_local))