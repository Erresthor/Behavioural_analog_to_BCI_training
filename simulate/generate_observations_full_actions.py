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

# We define the environment as a state machine that outputs a feedback 
# every time an action is given to it : 
from actynf.jaxtynf.layer_process import initial_state_and_obs,process_update
from actynf.jaxtynf.shape_tools import vectorize_weights

# A basic class to instantiate a POMDP generative process
# This serves as a vritual training environment for our 
# RL models and as a source of synthetic data for parameter / model retrieval tests

def initial_environment(rngkey,a,d):
    [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = initial_state_and_obs(rngkey,a,d)        
    return o_vect,jnp.array(0.0),s_vect

def step_environment(true_state,action_vectors,
                     previous_t,previous_obs,rngkey,
                     a,b):
    # action_chosen is a dictionnary of 3 action dimensions, but only 2 actually matter :         
    angle_action = action_vectors["angle"]
    distance_action = action_vectors["distance"]
    
    # The effective action is the one that will affect the system. It is driven 
    # by the angle, except when the distance between the points is too small :     
    na_action = jax.nn.one_hot(0,9) # No action recorded :
    effective_action = (1.0 - distance_action[0])*angle_action + distance_action[0]*na_action

    [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = process_update(rngkey,true_state,a,b,effective_action)
    N_outcomes = o_vect[0].shape[0]
    reward = (jnp.linspace(0,1,N_outcomes)*(o_vect[0] - previous_obs[0])).sum()
    
    return o_vect,reward,s_vect

class TrainingEnvironment :
    """
    Simulate a simple training environment based on a POMDP.    
    """
    def __init__(self,a,b,c,d,e,u,T,N_trials):
        # Environment parameters
        self.a = a
        self.b = b
        self.c = c 
        self.d = d
        self.e = e
        self.u = u
        self.T = T
        self.N_trials = N_trials
        
        self.vec_a,self.vec_b,self.vec_d = vectorize_weights(self.a,self.b,self.d,self.u)        
    
    def get_functions(self):
        func_initial_state_and_obs = partial(initial_environment,a = self.vec_a,d = self.vec_d)
        func_step = partial(step_environment,a = self.vec_a,b = self.vec_b)
        return func_initial_state_and_obs,func_step


# A set of functions to run the various proposal models in a virtual environment and invert
# empirical data from each of these models :
def simulate_training(environment_functions,agent_functions,rngkey_simulation,
                      n_trials = 10, n_observations_per_trial = 11):
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
    # n_trials = environment.N_trials
    # n_observations_per_trial = environment.T  # 1 more observation than action!
    func_env_initial_state,func_env_step = environment_functions 
    func_agent_init_params,func_agent_init_state,func_agent_step,func_agent_learn,_,_ = agent_functions
            # Don't need predictor or encoder in this !
    

    def _scan_trial(_carry,_data_trial):
        _agent_params = _carry
        _rngkey_trial = _data_trial
        
        def __scan_timestep(__carry,__data_timestep):
            __env_state,__env_obs,__env_reward,__agent_state = __carry
            __t,__rngkey_timestep = __data_timestep
            
            __choice_rng_key, __rngkey_timestep = jr.split(__rngkey_timestep)
            __obs_agent = (__env_obs,__env_reward,__t)
            
            __new_agent_state,(_,_,__u_vec),__reporting_data = func_agent_step(__obs_agent,__agent_state,_agent_params,__choice_rng_key)
            
            __new_agent_perceived_actions = __u_vec
            
            # Environment tick
            __env_rng_key, __rngkey_timestep = jr.split(__rngkey_timestep)
            __new_env_obs,__new_env_reward,__new_env_state = func_env_step(__env_state,__u_vec,
                                                __t,__env_obs,__env_rng_key)
                        
            return (__new_env_state,__new_env_obs,__new_env_reward,__new_agent_state),(__new_env_state,__new_env_obs,__new_env_reward,__new_agent_state,__u_vec,__new_agent_perceived_actions,__reporting_data)
        
        _initial_state_rngkey,_rngkey_trial = jr.split(_rngkey_trial)
        _initial_obs,_inital_reward,_initial_state_env = func_env_initial_state(_initial_state_rngkey)
        
        _initial_state_agent = func_agent_init_state(_agent_params)
        _initial_carry = (_initial_state_env,_initial_obs,_inital_reward,_initial_state_agent)
        _data = (jnp.arange(n_observations_per_trial),jr.split(_rngkey_trial,n_observations_per_trial))
        
        _,(_env_states,_env_obs,_env_rewards,_agent_states,_agent_actions,_agent_perceived_actions,_reporting_data) = jax.lax.scan(__scan_timestep,_initial_carry,_data)
        
        # We generated one more environment timestep than needed ! Let's remove the supplementary one
        # And include the initial carry value in the results
        _stich_up = lambda x0,x : jnp.concatenate([jnp.expand_dims(x0,0),x[:-1]])
        
        _env_states = _stich_up(_initial_state_env, _env_states)
        _env_obs = tree_map(_stich_up,_initial_obs, _env_obs)
        _env_rewards = _stich_up(_inital_reward, _env_rewards)
        _agent_states = _agent_states
        # _agent_states = tree_map(lambda x,y : jnp.concatenate([x,y]),_initial_state_agent,_agent_states)
        
        # We also disregard the last predicted action : (the state was updated but no action was performed)
        _agent_actions = tree_map(lambda x : x[:-1],_agent_actions)
        _agent_perceived_actions = tree_map(lambda x : x[:-1],_agent_perceived_actions)
        
        _new_params,_other_reporting_data = func_agent_learn((_env_rewards,_env_obs,_agent_states,_agent_perceived_actions),_agent_params)
                
        return _new_params,((_env_states,_env_obs,_env_rewards),(_agent_states,_new_params,_agent_actions,_reporting_data))


    # The initial parameters of the tested model are initialized once per training
    initial_parameters = func_agent_init_params()  
    data = (jr.split(rngkey_simulation,n_trials))
    final_parameters,(environment_variables,agent_variables) = jax.lax.scan(_scan_trial,initial_parameters,data)

    return final_parameters,(environment_variables,agent_variables)


def generate_synthetic_data(environment_object,agent_functions,n_trials=10,seed=0,verbose=True):
    # Synthetic data (here, generated randomly) :
    jr_key = jr.PRNGKey(seed)
    final_parameters,(environment_variables,agent_variables) = simulate_training(environment_object.get_functions(),agent_functions,
                  jr_key,n_trials=n_trials,n_observations_per_trial= environment_object.T)
    
    (_env_states,_env_obs,_env_rewards) = (environment_variables)
    (agent_states,agent_actions) = agent_variables
            
            
    formatted_stimuli= _env_obs
    bool_stimuli = [jnp.ones_like(stim[...,0]) for stim in formatted_stimuli]
    rewards = _env_rewards
    
    # Note the change here : 
    actions = agent_actions
    tmtsp = jnp.repeat(jnp.expand_dims(jnp.arange(environment_object.T),0),n_trials,0)
    
    # jnp.array(training_hist["timestamps"])
    synthetic_data = (formatted_stimuli,bool_stimuli,rewards,actions,tmtsp)
    
    return synthetic_data