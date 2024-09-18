import pymongo
import sys,os

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

def get_complete_collection():
    return get_ith_collection(0)
    
def get_partial_collection():
    return get_ith_collection(1)




if __name__ == "__main__":
    coll = get_complete_collection()
    for x in coll:
        print(x)
    print(coll)