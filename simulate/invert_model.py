import jax
import jax.random as jr
from jax import vmap

import jax.numpy as jnp
from functools import partial

from .compute_likelihood_full_actions import compute_loglikelihood,fit_mle_agent,fit_map_agent
from .models_utils import tree_stack

def invert_data_for_single_model(data_all_subjects,model_contents,
                                standard_n_heads = 20,standard_n_steps = 500,lr = 5e-2,
                                rngkey = jr.PRNGKey(0),option_verbose = True):
    formatted_stimuli,_,_,_,_ = data_all_subjects
    Nsubj,Ntrials,Ntimesteps,_ = formatted_stimuli[0].shape
    
    
    agent = model_contents["model"]
    initial_parameter_range = model_contents["ranges"]
    agent_priors = model_contents["priors"]
    
    # Model specific options :
    model_steps = model_contents["n_steps"] if ("n_steps" in model_contents) else standard_n_steps
    model_heads = model_contents["n_heads"] if ("n_heads" in model_contents) else standard_n_heads
    bypass_fit = model_contents["bypass"] if ("bypass" in model_contents) else False
    vectorize_computations = model_contents["vectorize_fit"] if ("vectorize_fit" in model_contents) else True
    
    
    # Agent functions
    _,_,_,_,_,encoder = agent(None)
    def fit_agent(_data_one_subject,_fit_rng_key,method = "mle"):  
        if bypass_fit:
            _opt_vectors = jnp.zeros((model_heads,))
            _lls =   vmap(lambda x : compute_loglikelihood(_data_one_subject,agent(encoder(x)),'sum'))(_opt_vectors)
            return None,None,_lls
            
            
        if method.lower() == "mle":
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_mle_agent(_data_one_subject,agent,
                                                                        initial_parameter_range,
                                                                        _fit_rng_key,
                                                                        true_hyperparams=None,                          
                                                                        num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,
                                                                        verbose=option_verbose)
        
        elif method.lower() == "map":
            # Multi-iteration based MAP : (we randomize the initial point and try to find minimas)
            _opt_vectors,(_,_loss_history,_param_history),_encoding_function = fit_map_agent(_data_one_subject,agent,
                                                                        initial_parameter_range,agent_priors,
                                                                        _fit_rng_key,
                                                                        true_hyperparams=None,                          
                                                                        num_steps=model_steps,n_heads=model_heads,
                                                                        start_learning_rate=lr,
                                                                        verbose=option_verbose)
        
        else :            
            raise NotImplementedError("Unrecognized fitting method : {}".format(method))
        
        # Regardless of the cost function, let's estimate the log-likelihood of the fitted estimators :
        _lls =   vmap(lambda x : compute_loglikelihood(_data_one_subject,agent(_encoding_function(x)),'sum'))(_opt_vectors)
        return _loss_history,_opt_vectors,_lls
    
    
    
    if vectorize_computations : # Better, as long as we don't run into memory issues
        fit_one_subject = partial(fit_agent,_fit_rng_key=rngkey)
        loss_histories,best_params,lls = vmap(fit_one_subject)(data_all_subjects)
        
        fit_results = {
            "losses_hist" : loss_histories,
            "params" : best_params,
            "logliks" : lls,
            "encoder" : encoder
        }
    else : 
        random_keys_all_subj = jr.split(rngkey,num = Nsubj)
        
        clean_this = []
        for subj in range(Nsubj):
            print("Subject {}/{}".format(subj,Nsubj))
            
            subj_key = random_keys_all_subj[subj]
            
            # Extract the subj-th element of each array :
            (formatted_stimuli,bool_stimuli,rewards,actions,timesteps) = data_all_subjects
            
            data_this_subject = (formatted_stimuli[0][subj],bool_stimuli[0][subj],rewards[subj],
                                {dim : table[subj] for dim,table in actions.items()},timesteps[subj])
            
            loss_hist, opt_vec, lls = fit_agent(data_this_subject,subj_key)
            clean_this.append({
                "losses_hist" : loss_hist,
                "params" : opt_vec,
                "logliks" : lls,
                "encoder" : encoder
            })
            
        # Clean the list here
        def cleaner(_list):
            new_dict = {key:tree_stack([el[key] for el in _list]) for key in ["losses_hist","params","logliks"]}
            new_dict["encoder"] = _list[0]["encoder"]
            return new_dict
        
        fit_results = cleaner(clean_this)
        
    return fit_results
 
    # results_dict[agent_name] = fit_results

def invert_data_for_library_of_models(data_all_subjects,model_library,
                                      standard_n_heads = 20,standard_n_steps = 500,lr = 5e-2,
                                      rngkey = jr.PRNGKey(0),option_verbose = True):

    results_dict = {}
    for agent_name, agent_contents in model_library.items():
        print("     -> Agent : {}".format(agent_name))
        rngkey,local_key = jr.split(rngkey)
        
        results_dict[agent_name] = invert_data_for_single_model(data_all_subjects,agent_contents,
                                                                standard_n_heads = standard_n_heads,standard_n_steps = standard_n_steps,lr = lr,
                                                                rngkey = local_key,option_verbose = option_verbose)
    
    return results_dict