import jax
import jax.random as jr
from jax import vmap

import jax.numpy as jnp
from functools import partial

from .compute_likelihood_full_actions import compute_loglikelihood,fit_mle_agent,fit_map_agent
from .simulate_utils import tree_stack

import os
import pickle


def invert_data_for_single_model(data_all_subjects,model_contents,method="mle",
                                standard_n_heads = 20,standard_n_steps = 500,lr = 5e-2,lr_scheduler=None,
                                rngkey = jr.PRNGKey(0),option_verbose = True,
                                save=False ,save_directory = "default",override = False):
    
    if os.path.exists(save_directory):
        if not override:
            with open(save_directory, 'rb') as inp:
                existing_results = pickle.load(inp)
            return existing_results
    
    
    formatted_stimuli,_,_,_,_ = data_all_subjects
    Nsubj,Ntrials,Ntimesteps,_ = formatted_stimuli[0].shape
    
    
    agent_object = model_contents["agent"]
    encoder = agent_object.get_encoder()
    # initial_parameter_range = agent_object.get_initial_ranges()
    # agent_priors = model_contents["priors"] if ("priors" in model_contents) else agent_object.get_priors()
    
    _local_rngkey,rngkey = jr.split(rngkey)
    total_infered_vals,_ = jax.tree.flatten(jax.tree_util.tree_map(lambda x : x.size,agent_object.get_random_parameters(_local_rngkey)))
    total_infered_vals = sum(total_infered_vals)
    print(total_infered_vals)
    
    
    # Model specific options :
    model_steps = model_contents["n_steps"] if ("n_steps" in model_contents) else standard_n_steps
    model_heads = model_contents["n_heads"] if ("n_heads" in model_contents) else standard_n_heads
    bypass_fit = model_contents["bypass"] if ("bypass" in model_contents) else False
    vectorize_computations = model_contents["vectorize_fit"] if ("vectorize_fit" in model_contents) else True
    
    
    # Agent functions
    def fit_agent(_data_one_subject,_fit_rng_key):  
        if (bypass_fit | (total_infered_vals==0)):
            _opt_vectors = jnp.zeros((model_heads,))
            _lls =   vmap(lambda x : compute_loglikelihood(_data_one_subject,agent_object.get_all_functions(encoder(x)),'sum'))(_opt_vectors)
            return None,None,_lls
        
        if method.lower() == "mle":
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_mle_agent(_data_one_subject,agent_object,_fit_rng_key,
                                                                        true_hyperparams=None, num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,lr_schedule_dict=lr_scheduler,
                                                                        verbose=option_verbose)
        
        elif method.lower() == "map":
            # Multi-iteration based MAP : (we randomize the initial point and try to find minimas)
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_map_agent(_data_one_subject,agent_object,_fit_rng_key,
                                                                        true_hyperparams=None, num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,lr_schedule_dict=lr_scheduler, 
                                                                        verbose=option_verbose)
        
        else :            
            raise NotImplementedError("Unrecognized fitting method : {}".format(method))
        
        # Regardless of the cost function, let's estimate the log-likelihood of the fitted estimators :
        _lls =   vmap(lambda x : compute_loglikelihood(_data_one_subject,agent_object.get_all_functions(_encoding_function(x)),'sum'))(_opt_vectors)
        return _loss_history,_opt_vectors,_lls
    
    
    
    if vectorize_computations : # Usually better, as long as we don't run into memory issues or long compilation times
        fit_one_subject = partial(fit_agent,_fit_rng_key=rngkey)
        loss_histories,best_params,lls = vmap(fit_one_subject)(data_all_subjects)
        
        fit_results = {
            "losses_hist" : loss_histories,
            "params" : best_params,
            "logliks" : lls
        }
    else : 
        random_keys_all_subj = jr.split(rngkey,num = Nsubj)
        
        clean_this = []
        for subj in range(Nsubj):
            print("Subject {}/{}".format(subj,Nsubj))
            
            subj_key = random_keys_all_subj[subj]
            
            # Extract the subj-th element of each array :
            (formatted_stimuli,bool_stimuli,rewards,actions,timesteps) = data_all_subjects
            
            data_this_subject = ([formatted_stimuli[0][subj]],[bool_stimuli[0][subj]],rewards[subj],
                                {dim : table[subj] for dim,table in actions.items()},timesteps[subj])
            
            loss_hist, opt_vec, lls = fit_agent(data_this_subject,subj_key)
            clean_this.append({
                "losses_hist" : loss_hist,
                "params" : opt_vec,
                "logliks" : lls
            })
            
        # Clean the list here
        def cleaner(_list):
            new_dict = {key:tree_stack([el[key] for el in _list]) for key in ["losses_hist","params","logliks"]}
            # new_dict["encoder"] = _list[0]["encoder"]
            return new_dict
        
        fit_results = cleaner(clean_this)
    
    if save :
        with open(save_directory, 'wb') as outp:
            pickle.dump(fit_results, outp,pickle.HIGHEST_PROTOCOL)
    
    return fit_results
 

def invert_data_for_single_model_cluster(executor,joblist,
                                data_all_subjects,model_contents,method="mle",
                                standard_n_heads = 20,standard_n_steps = 500,lr = 5e-2,lr_scheduler=None,
                                rngkey = jr.PRNGKey(0),option_verbose = True,
                                save=False ,save_directory = "default",override = False):
    import submitit    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    formatted_stimuli,_,_,_,_ = data_all_subjects
    Nsubj,Ntrials,Ntimesteps,_ = formatted_stimuli[0].shape
    
    
    agent_object = model_contents["agent"]
    encoder = agent_object.get_encoder()
    
    _local_rngkey,rngkey = jr.split(rngkey)
    total_infered_vals,_ = jax.tree.flatten(jax.tree_util.tree_map(lambda x : x.size,agent_object.get_random_parameters(_local_rngkey)))
    total_infered_vals = sum(total_infered_vals)
    
    # Model specific options :
    model_steps = model_contents["n_steps"] if ("n_steps" in model_contents) else standard_n_steps
    model_heads = model_contents["n_heads"] if ("n_heads" in model_contents) else standard_n_heads
    
    # Agent functions
    def fit_agent(_data_one_subject,_fit_rng_key):  
        if method.lower() == "mle":
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_mle_agent(_data_one_subject,agent_object,_fit_rng_key,
                                                                        true_hyperparams=None, num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,lr_schedule_dict=lr_scheduler,
                                                                        verbose=option_verbose)
        
        elif method.lower() == "map":
            # Multi-iteration based MAP : (we randomize the initial point and try to find minimas)
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_map_agent(_data_one_subject,agent_object,_fit_rng_key,
                                                                        true_hyperparams=None, num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,lr_schedule_dict=lr_scheduler, 
                                                                        verbose=option_verbose)
        else :            
            raise NotImplementedError("Unrecognized fitting method : {}".format(method))
        # Regardless of the cost function, let's estimate the log-likelihood of the fitted estimators :
        _lls =   vmap(lambda x : compute_loglikelihood(_data_one_subject,agent_object.get_all_functions(_encoding_function(x)),'sum'))(_opt_vectors)
        return _loss_history,_opt_vectors,_lls
    
    def fit_one_subject(_subject_save_directory,_data_this_subject,_rng_key_subj):
        if os.path.exists(_subject_save_directory):
            if not override:
                with open(_subject_save_directory, 'rb') as inp:
                    existing_results = pickle.load(inp)
                return existing_results
        
        loss_hist, opt_vec, lls = fit_agent(_data_this_subject,_rng_key_subj)
        fit_results = {
            "losses_hist" : loss_hist,
            "params" : opt_vec,
            "logliks" : lls
        }
        
        if save :
            with open(_subject_save_directory, 'wb') as outp:
                pickle.dump(fit_results, outp,pickle.HIGHEST_PROTOCOL)
            
        return fit_results
    
    # Extract the subj-th element of each array :
    (formatted_stimuli,bool_stimuli,rewards,actions,timesteps) = data_all_subjects
        
    # Do subjects one at a time :        
    random_keys_all_subj = jr.split(rngkey,num = Nsubj)
    full_results = []
    with executor.batch():           
        for subj in range(Nsubj):
            print("Subject {}/{}".format(subj,Nsubj))
            subject_save_directory = os.path.join(save_directory,"subj_{}".format(subj))
            subj_key = random_keys_all_subj[subj]            
            data_this_subject = ([formatted_stimuli[0][subj]],[bool_stimuli[0][subj]],rewards[subj],
                                {dim : table[subj] for dim,table in actions.items()},timesteps[subj])
                        
            job = executor.submit(fit_one_subject, subject_save_directory,data_this_subject,subj_key)
            joblist.append(job)
        
    return joblist
 
 
 
 
 
 
def invert_data_for_library_of_models(data_all_subjects,model_library,method="mle",
                                      standard_n_heads = 20,standard_n_steps = 500,lr = 5e-2,lr_scheduler=None,
                                      rngkey = jr.PRNGKey(0),option_verbose = True,
                                      save=False ,save_directory = "default",override = False):
    if save :
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    results_dict = {}
    for agent_name, agent_contents in model_library.items():
        print("     -> Agent : {}".format(agent_name))
        
        model_path = os.path.join(save_directory,agent_name)
        
        
        rngkey,local_key = jr.split(rngkey)
        
        results_dict[agent_name] = invert_data_for_single_model(data_all_subjects,agent_contents,method=method,
                                                                standard_n_heads = standard_n_heads,standard_n_steps = standard_n_steps,
                                                                lr = lr,lr_scheduler=lr_scheduler,
                                                                rngkey = local_key,option_verbose = option_verbose,
                                                                save=save,save_directory=model_path,override=override)       
        
    return results_dict