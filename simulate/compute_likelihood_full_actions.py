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



def _uniform_sample_leaf(_rng_leaf,_range_leaf, size):
    """
    Given a jr.PRNGKey and a (2,)-shaped tensor of lower and upper bound, 
    return a randpm tensor of size "size" sampled from U(lb,ub)

    Args:
        _rng_leaf (_type_): _description_
        _range_leaf (_type_): _description_
        size (_type_): _description_

    Returns:
        _type_: _description_
    """
    return jr.uniform(_rng_leaf,(size,),minval  = _range_leaf[0], maxval =_range_leaf[1])





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
            
            return __new_state,(__predicted_action,__new_state,__other_data)
        
        
        
        _,(_predicted_actions,_trial_states,_trial_other_data) = jax.lax.scan(__scan_timestep, (_initial_state),_expanded_data_trial)
        
        _new_params = agent_learn((_rewards_trial,_observations_trial,_trial_states,_actions_trial),_agent_params)
        
        _removed_last_predicted_action = tree_map(lambda x : x[:-1,...],_predicted_actions)
        
        return _new_params,(_removed_last_predicted_action,(_trial_states,_trial_other_data))

    final_parameters,(predicted_actions,(model_states,other_data)) = jax.lax.scan(_scan_trial,initial_parameters,data)

    return final_parameters,predicted_actions,(model_states,other_data)


def compute_loglikelihood(data,agent_functions,statistic='mean',return_params=False):   
    final_parameters,predicted_actions,(model_states,other_data) = compute_predicted_actions(data,agent_functions)
    
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
    
    if return_params:
        return (logliks_dict,ce_dict),(predicted_actions,final_parameters),(model_states,other_data)
    
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


def fit_mle_agent(data,static_agent,feature_initial_range,rngkey,
            true_hyperparams=None,n_heads=10,num_steps=100,
            start_learning_rate = 1e-1,verbose=False,):
    """This REALLY should have been done with a class ..."""
    
    # The initial param encoding function : 
    _,_,_,_,_,encoding_function = static_agent(None)
    
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
    sampler = partial(_uniform_sample_leaf,size=n_heads)
    initial_feature_vectors = tree_map(sampler,rng_key_tree,feature_initial_range)


    # Gradient descent on the MLE :
    optimizer = optax.adam(start_learning_rate)
    fit_this = partial(fit,obs=data,loss_func = mle_loss,optimizer=optimizer,num_steps = num_steps,param_history=True,verbose=verbose)

    all_fin_params,all_losses,all_param_histories = vmap(fit_this)(initial_feature_vectors)
    
    return all_fin_params,(gt_mle,all_losses,all_param_histories),encoding_function



# MAP fitting methods : 
def compute_log_prob(_it_param,_it_prior_dist):
    _mapped = tree_map(lambda x,y : y.log_prob(x),_it_param,_it_prior_dist)
    
    if isinstance(_mapped,dict):
        _mapped = list(_mapped.values())
    
    _params_lp = jnp.stack(_mapped)
    return jnp.sum(_params_lp),_params_lp


def fit_map_agent(data,static_agent,
            N_hyperparams,priors,
            rngkey,
            true_hyperparams=None,n_iter=10,num_steps=100,
            initial_window = [-10,10],verbose=False,
            ll_statistic="sum"):
    """This REALLY should have been done with a class ..."""
    
    if (priors is not None):
        # For the MAP estimate, we assume a Mean Field approximation between the parameters of our models
        # More complex hypothesis can be entertained when doing Bayesian Parameter Estimation (MCMC / SVI) using Pyro models 
        assert len(priors)==N_hyperparams,"There should be as many prior distributions as there are parameters ({})".format(N_hyperparams)
    
    log_prior_func = partial(compute_log_prob,_it_prior_dist = priors)
    
    # The initial param encoding function : 
    _,_,_,_,_,encoding_function = static_agent(None)

    
    def log_posterior(_hyperparameters,_data):
        
        log_prior,_ = log_prior_func(_hyperparameters)
        
        log_likelihood = compute_loglikelihood(_data,static_agent(_hyperparameters),ll_statistic)
        
        return (log_likelihood + log_prior)
    
    
    # Minimize this :
    def generic_loss(_X,_observed_data):
        
        _hyperparameters = encoding_function(_X)
        
        return - log_posterior(_hyperparameters,_observed_data) 
    
    
    if not(true_hyperparams is None):
        # MLE Value of the true parameters : 
        gt_map = - log_posterior(true_hyperparams,data)
    else :
        gt_map = None
    
    # Gradient descent on the MLE :
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)
    
    # Grab a few initial starting positions
    candidates = jr.uniform(rngkey,(n_iter,N_hyperparams),minval  = initial_window[0], maxval = initial_window[1])

    fit_this = partial(fit,obs=data,loss_func = generic_loss,optimizer=optimizer,num_steps = num_steps,param_history=True,verbose=verbose)

    all_fin_params,all_losses,all_param_histories = vmap(fit_this)(candidates)

    loss_history = jnp.stack(all_losses)
    
    return all_fin_params,(gt_map,loss_history,all_param_histories),encoding_function