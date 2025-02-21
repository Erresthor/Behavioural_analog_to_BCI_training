# Import the needed packages 
# 
# 1/ the usual suspects
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map
import optax

from functools import partial

# 2/ The Active Inference package 
import actynf
from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog
from actynf.jaxtynf.jax_toolbox import random_split_like_tree

from .simulate_utils import uniform_sample_leaf

# Likelihood computations and MLE + MAP fitting algorithms for a dictionnary structure
def get_random_parameter_set(feature_initial_range,rngkey,n_heads = 1,autosqueeze = False):
    # Grab a few initial starting positions
    rng_key_tree = random_split_like_tree(rngkey,feature_initial_range)
    
    sampler = partial(uniform_sample_leaf,size=n_heads)        
            
    initial_feature_vectors = tree_map(sampler,rng_key_tree,feature_initial_range)
    
    if autosqueeze:
        return jax.tree_map(lambda x : jnp.squeeze(x,axis=0),initial_feature_vectors)
    return initial_feature_vectors


def compute_predicted_actions(data,agent_functions):
    """A function that uses vmap to compute the predicted agent action at time $t$ given $o_{1:t}$ and $u_{1:t-1}$. 
    This function should be differentiable w.r.t. the hyperparameters of the agent's model because we're going to perform
    gradient descent on it !

    Args:
        environment (_type_): _description_
        agent_functions (_type_): _description_
        seed (_type_): _description_
        Ntrials (_type_): _description_

    Returns:
        _type_: _description_
        
        
    Note : the high memory usage of this function is partly due to it storing the full model states across all timepoints and returning them. 
    We should make a simple function with minimum memory usage for fitting and a bigger one with full reporting (states AND parameters) to monitor
    probable states.
    """
    init_params,init_state,_,agent_learn,predict,_ = agent_functions
    
    
    # Data should contain :
    # - all observations -> stimuli,reward (from the system)
    #       -> a list of stimuli for each modality
    #       -> a list of observation filters for each modality
    #       -> a Ntrials x Ntimesteps tensor array of scalar rewards (\in [0,1])
    # - all true actions 
    #       -> a Ntrials x (Ntimesteps-1) x Nu tensor array encoding the observed actions
    #       -> a Ntrials x (Ntimesteps-1) filter tensor indicating which actions were NOT observed
    
    initial_parameters = init_params()  
        # The initial parameters of the tested model are initialized once per training
    
    
    def _scan_trial(_carry,_data_trial):
        
        _agent_params = _carry
        _initial_state = init_state(_agent_params)
        
        _observations_trial,_observations_filter_trial,_rewards_trial,_actions_trial,_timestamps_trial = _data_trial
        
        # The same actions, with an extra one at the end for scan to work better !
        _expanded_actions_trial = tree_map(lambda x : jnp.concatenate([x,jnp.zeros((1,x.shape[-1]))]),_actions_trial)
        _expanded_data_trial = (_observations_trial,_observations_filter_trial,_rewards_trial,_expanded_actions_trial,_timestamps_trial)
        
        def __scan_timestep(__carry,__data_timestep):
            __agent_state = __carry
            
            __new_state,__predicted_action,__other_data = predict(__data_timestep,__agent_state,_agent_params)        
            
            (_,_,_,__perceived_action_timestep,_) = __data_timestep

            return __new_state,(__predicted_action,__perceived_action_timestep,__new_state,__other_data)
        
        _,(_predicted_actions,_perceived_actions,_trial_states,_trial_other_data) = jax.lax.scan(__scan_timestep, (_initial_state),_expanded_data_trial)
        _removed_last_predicted_action = tree_map(lambda x : x[:-1,...],_predicted_actions)
        _removed_last_perceived_actions = tree_map(lambda x : x[:-1,...],_perceived_actions)
                
        _new_params,_other_reporting_data = agent_learn((_rewards_trial,_observations_trial,_trial_states,_removed_last_perceived_actions),_agent_params)
        
        _reporting_dict = {**_trial_other_data,**_other_reporting_data}
        
        return _new_params,(_removed_last_predicted_action,(_trial_states,_new_params,_reporting_dict))

    final_parameters,(predicted_actions,(model_states,model_params,other_data)) = jax.lax.scan(_scan_trial,initial_parameters,data)

    return final_parameters,predicted_actions,(model_states,model_params,other_data)


def compute_predicted_actions_basic(data,agent_functions):
    """A function that uses vmap to compute the predicted agent action at time $t$ given $o_{1:t}$ and $u_{1:t-1}$. 
    This function should be differentiable w.r.t. the hyperparameters of the agent's model because we're going to perform
    gradient descent on it !

    Args:
        environment (_type_): _description_
        agent_functions (_type_): _description_
        seed (_type_): _description_
        Ntrials (_type_): _description_

    Returns:
        _type_: _description_
        
        
    Note : A lower memory usage than compute_predicted_actions, making it more adapted to batch gradient descent.
    """
    init_params,init_state,_,agent_learn,predict,_ = agent_functions
    
    
    # Data should contain :
    # - all observations -> stimuli,reward (from the system)
    #       -> a list of stimuli for each modality
    #       -> a list of observation filters for each modality
    #       -> a Ntrials x Ntimesteps tensor array of scalar rewards (\in [0,1])
    # - all true actions 
    #       -> a Ntrials x (Ntimesteps-1) x Nu tensor array encoding the observed actions
    #       -> a Ntrials x (Ntimesteps-1) filter tensor indicating which actions were NOT observed
    
    initial_parameters = init_params()  
        # The initial parameters of the tested model are initialized once per training
    
    
    def _scan_trial(_carry,_data_trial):
        
        _agent_params = _carry
        _initial_state = init_state(_agent_params)
        
        _observations_trial,_observations_filter_trial,_rewards_trial,_actions_trial,_timestamps_trial = _data_trial
        
        # The same actions, with an extra one at the end for scan to work better !
        _expanded_actions_trial = tree_map(lambda x : jnp.concatenate([x,jnp.zeros((1,x.shape[-1]))]),_actions_trial)
        _expanded_data_trial = (_observations_trial,_observations_filter_trial,_rewards_trial,_expanded_actions_trial,_timestamps_trial)
        
        def __scan_timestep(__carry,__data_timestep):
            __agent_state = __carry
            
            __new_state,__predicted_action,__other_data = predict(__data_timestep,__agent_state,_agent_params)        
            
            (_,_,_,__perceived_action_timestep,_) = __data_timestep
            
            return __new_state,(__predicted_action,__perceived_action_timestep,__new_state,__other_data)
        
        _,(_predicted_actions,_perceived_actions,_trial_states,_) = jax.lax.scan(__scan_timestep, (_initial_state),_expanded_data_trial)
        _removed_last_predicted_action = tree_map(lambda x : x[:-1,...],_predicted_actions)
        _removed_last_perceived_actions = tree_map(lambda x : x[:-1,...],_perceived_actions)
                
        _new_params,_ = agent_learn((_rewards_trial,_observations_trial,_trial_states,_removed_last_perceived_actions),_agent_params)
                
        return _new_params,_removed_last_predicted_action

    final_parameters,predicted_actions = jax.lax.scan(_scan_trial,initial_parameters,data)

    return final_parameters,predicted_actions




def compute_loglikelihood(data,agent_functions,statistic='mean',full_report=False):   
    
    final_parameters,predicted_actions = compute_predicted_actions(data,agent_functions)
    
    # This function will be mapped to all action modalities :
    def _cross_entropy_action_modality(_true_action,_predicted_action):
        # Cross entropy (observed vs predicted) :
        _ce = (_true_action * _jaxlog(_predicted_action)).sum(axis=-1)
        return _ce
    
    def _loglik_action_modality(_cross_entropy):
        # Here's the average log-likelihood of what was observed given this model :
        if statistic == "mean":
            return jnp.mean(_cross_entropy)
        elif statistic == "sum":
            return jnp.sum(_cross_entropy)
        else : 
            raise NotImplementedError("Unimplemented statistic : {}".format(statistic))

    (_,_,_,actions,_) = data  # Get the true actions to qualify how good our model is :

    ce_dict = tree_map(_cross_entropy_action_modality,actions,predicted_actions)
    logliks_dict = tree_map(_loglik_action_modality,ce_dict)
    
    if full_report:
        return (logliks_dict,ce_dict),(predicted_actions,final_parameters)
    
    return logliks_dict,ce_dict


# A generic fitting function through SGD
def fit(params, obs, loss_func, optimizer, num_steps = 100, 
        verbose=False,param_history=False):
    """ I'm fast as fk boi """
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_func)(params, obs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    list_params = [params]
    for i in range(num_steps):
        params, opt_state, loss_value = step(params, opt_state)
        losses.append(loss_value)
        if param_history:
            list_params.append(params)
        
        if verbose and (i % 10 == 0):
            print(f'step {i}, loss: {loss_value}')
    
    if param_history:
        # Transpose list of dicts
        list_params = jax.tree.map(lambda *xs: jnp.stack(list(xs)), *list_params)
    
    return params,jnp.array(losses),list_params


def fit_mle_agent(data,agent_object,rngkey,
            true_hyperparams=None,n_heads=10,num_steps=100,
            start_learning_rate = 1e-1,lr_schedule_dict = None,
            verbose=False):
    
    # The initial param encoding function : 
    encoding_function = agent_object.get_encoder()
    feature_initial_range = agent_object.get_initial_ranges()
    static_agent = partial(agent_object.get_all_functions)
    
    
     # The loss function we use :
    def mle_loss(_X,_observed_data):
        _hyperparameters = encoding_function(_X)
        lls_tree,_ = compute_loglikelihood(_observed_data,static_agent(_hyperparameters),"sum")
        return - jax.tree_util.tree_reduce(lambda x,y : x+y,lls_tree)
    
    
    if not(true_hyperparams is None):
        # MLE Value of the true parameters : 
        gt_mle = - mle_loss(true_hyperparams,data)
    else :
        gt_mle = None
    
    # Grab a few initial starting positions
    rng_key_tree = random_split_like_tree(rngkey,feature_initial_range)
    sampler = partial(uniform_sample_leaf,size=n_heads)
    initial_feature_vectors = tree_map(sampler,rng_key_tree,feature_initial_range)


    # Gradient descent on the log-likelihood :
    if lr_schedule_dict is None : 
        lr_schedule_dict = {1000: start_learning_rate/2.0, 5000: start_learning_rate/10.0} # Change at step 1000 and 5000 
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=start_learning_rate,
        boundaries_and_scales=  lr_schedule_dict 
    )
    optimizer = optax.adam(lr_schedule)
    fit_this = partial(fit,obs=data,loss_func = mle_loss,optimizer=optimizer,num_steps = num_steps,param_history=True,verbose=verbose)

    all_fin_params,all_losses,all_param_histories = vmap(fit_this)(initial_feature_vectors)
    
    return all_fin_params,(gt_mle,all_losses,all_param_histories),encoding_function


# _____________________________________________________________________________
# MAP fitting methods : 
def compute_log_prob(_it_param,_it_prior_dist):
    _mapped = tree_map(lambda x,y : jnp.sum(y.log_prob(x)),_it_param,_it_prior_dist)
        # Added sum here to account for vector parameters
    return jax.tree_util.tree_reduce(lambda x,y : x+y,_mapped),_mapped


def fit_map_agent(data,agent_object,rngkey,
            true_hyperparams=None,n_heads=10,num_steps=100,
            start_learning_rate = 1e-1,lr_schedule_dict = None,
            verbose=False):
    
    encoding_function = agent_object.get_encoder()
    feature_initial_range = agent_object.get_initial_ranges()
    priors = agent_object.get_priors()
    static_agent = partial(agent_object.get_all_functions)
    
    log_prior_func = partial(compute_log_prob,_it_prior_dist = priors)
    
    def log_posterior(_hyperparameters,_observed_data):
        log_prior,_ = log_prior_func(_hyperparameters)
        
        lls_tree,_ = compute_loglikelihood(_observed_data,static_agent(_hyperparameters),"sum") # One per action modality
        log_likelihood = jax.tree_util.tree_reduce(lambda x,y : x+y,lls_tree)  # Sum them ! 
        
        return (log_likelihood + log_prior)
    
    # Minimize the negative log posterior :
    def map_loss(_X,_observed_data):
        _hyperparameters = encoding_function(_X)
        return - log_posterior(_hyperparameters,_observed_data) 
    
    
    if not(true_hyperparams is None):
        # MAP Value of the true parameters : 
        gt_map = - map_loss(true_hyperparams,data)
    else :
        gt_map = None
    
    
    
    # Grab a few initial starting positions in feature space (ideally, this would be done directly in parameter space, 
    # but it would require a decoder for all models and I can't be bothered)
    rng_key_tree = random_split_like_tree(rngkey,feature_initial_range)
    sampler = partial(uniform_sample_leaf,size=n_heads)
    initial_feature_vectors = tree_map(sampler,rng_key_tree,feature_initial_range)
    
    if lr_schedule_dict is None : 
        lr_schedule_dict = {1000: start_learning_rate/2.0, 5000: start_learning_rate/10.0} # Change at step 1000 and 5000 
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=start_learning_rate,
        boundaries_and_scales=  lr_schedule_dict 
    )
    optimizer = optax.adam(lr_schedule)
    fit_this = partial(fit,obs=data,loss_func = map_loss,optimizer=optimizer,num_steps = num_steps,param_history=True,verbose=verbose)

    all_fin_params,all_losses,all_param_histories = vmap(fit_this)(initial_feature_vectors)
    
    return all_fin_params,(gt_map,all_losses,all_param_histories),encoding_function

