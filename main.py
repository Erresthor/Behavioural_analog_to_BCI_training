

from database_handling.database_extract import get_all_subject_data_from_internal_task_id

if __name__=="__main__":   
            
    internal_task_id = '002'
    r = get_all_subject_data_from_internal_task_id(internal_task_id,
                                None,
                                autosave=True,override_save=False,autoload=True)
    print(r)
    exit()