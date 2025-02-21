import pymongo
import sys,os
import json

LOCAL_SAVE_PATH = os.path.join("ressources","local_experiment_saves","local_database.json")

def load_client_address(filename="client_address.private"):
    private_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),"ressources", filename)
    return open(private_file,'r').read().split('\n')

def get_ith_collection(i):
    try :
        [client,database_name,collection_name] = load_client_address()[i].split(" ")
    except:
        raise ValueError("No collection with index '" + str(i) + "'.")
    
    # 1. Get the completed recordings from Atlas MongoDB :
    mongodb_client = pymongo.MongoClient(client)
    db = mongodb_client[database_name]
    coll = db[collection_name] # All the data from the subjects we're interested in !
    return coll


def save_collection_locally(collec,filepath = LOCAL_SAVE_PATH):
    if os.path.exists(filepath):
        print(f"File '{filepath}' already exists.")  # Or prompt for action
        return
      
    # 1. Get the completed recordings from Atlas MongoDB :
    collection_complete = collec.find()
    
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


def get_complete_collection(use_local_data = True,save_if_absent = True):
    
    if use_local_data:
        if os.path.exists(LOCAL_SAVE_PATH):
            return load_local_data_into_pymongo_collection(LOCAL_SAVE_PATH)
        else :
            print("Did not find any local data at " + LOCAL_SAVE_PATH)


    complete_collection_list = get_ith_collection(0).find()
    
    if save_if_absent:
        save_collection_locally(complete_collection_list)

    return complete_collection_list
    
def get_partial_collection():
    return get_ith_collection(1).find()



if __name__ == "__main__":
    coll = get_complete_collection()
    for x in coll:
        print(x)
    print(coll)