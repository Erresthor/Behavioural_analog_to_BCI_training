import pathlib
import pickle 

# ChatGPT sourced : 
def compare_dicts(dict1, dict2, get_exception = False):
    if dict1.keys() != dict2.keys():
        return False, "Key mismatch : {} vs {}".format(dict1.keys(), dict2.keys())
    
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            
            child_dict_bool, child_dict_code = compare_dicts(dict1[key], dict2[key])
            if not child_dict_bool:
                return False, child_dict_code
        
        elif dict1[key] != dict2[key]:
            return False, "{} =/= {}".format(dict1[key],dict2[key])
    return True,None


# A set of methods to allow a LLM agent to classify the text answers of  our subjects 
def ask_llm(llm_client,filepath,question,all_tasks_results,
            message_template,
            override_existing = False):
    file_path = pathlib.Path(filepath)
    
    (question_code,question_contents) = question
    
    # Creat save folder if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True) 
    
    
    
    # Check for an existing dictionnary of answers or create it :
    if file_path.exists():
        with open(filepath, 'rb') as f:
            all_llms_results = pickle.load(f)
    else:
        all_llms_results = {}
    
    
    
    # Check if the question already exists in the dict : 
    if question_code in all_llms_results.keys():
        # Make sure this question is the exact same as the previous :
        if not override_existing :
            
            dict_are_same, dict_arent_same_explanation = compare_dicts(all_llms_results[question_code]["prompt"],question_contents)
            
            if not(dict_are_same):
                raise Exception("The question code {} already exists but has different contents : \n {}".format(question_code,dict_arent_same_explanation))
            
    else :
        all_llms_results[question_code]["Template"] = message_template(question_contents,"[SUBJECT ANSWER]")
        all_llms_results[question_code]["prompt"] = question_contents 
        all_llms_results[question_code]["results"] = {} 
    
    
    
    # For all tasks in the input data :
    for task_id,task_results in all_tasks_results.items():
        # Check if the question already exists in the dict : 
        if task_id in all_llms_results[question_code]["results"].keys():
            if not override_existing :
                print("Skipping {} for task {} (already exists).".format(question_code,task_id))
                continue # No need to ask the LLM, classification already exists !
            
        all_llms_results[question_code]["results"][task_id] = []
        
        n_subjects = len(task_results)
        for k,subject_results in enumerate(task_results):
            print("{}/{}".format(k+1,n_subjects))
            
            subject_dict,trial_data,events,fb_rtv = subject_results
            
            
            subject_answer = subject_dict[question["dict_key"]]

            message = message_template(question,subject_answer)

            completion = llm_client.chat.completions.create(
                model="model-identifier",
                messages=message,
                temperature = 0.1, # Low temperature for deterministic answers
            )

            detected_category = completion.choices[0].message.content
            
            all_llms_results[question_code]["results"][task_id].append(detected_category)
            
        
        # Update the file every time a task is analyzed to allow for interruptions / minimize
        # the impact of crashes, which are likely to occur with big models:
        # with open(saveto, 'rb') as f:
        #     loaded_dict = pickle.load(f)
        
        
        with open(file_path, 'wb') as f:
            pickle.dump(all_llms_results, f)
    
        # with file_path.open("wb") as file:
        #     pickle.dump(all_llms_results, file)
        print("Updated dictionary at {}.".format(file_path))