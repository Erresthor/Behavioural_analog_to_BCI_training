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

from functools import partial

# We define the environment as a state machine that outputs a feedback 
# every time an action is given to it : 
from actynf.jaxtynf.layer_process import initial_state_and_obs,process_update
from actynf.jaxtynf.shape_tools import vectorize_weights




# A set of functions to run the various proposal models in a virtual environment and invert
# empirical data from each of these models :

# run_loop is the forward mode : given an ageent and an environment, it will simulate 
# the behaviour of an agent and report its history :)
def run_loop(environment,agent_functions,seed,Ntrials,verbose=True):
    rng_key = jr.PRNGKey(seed)
    
    
    env_key,agent_key = jr.split(rng_key)
    environment.rng_key = env_key
    
    init_params,init_state,agent_step,agent_learn,_,_ = agent_functions
    
    params = init_params()
    
    full_history = {
        "rewards" : [],
        "stimuli" : [],
        "actions" : [],
        "timestamps" : [],
        "states" : [],
        "params" : [],
        "env_states" : [],
    }
    
    for trial in range(Ntrials):
        if verbose:
            print("Trial {}".format(trial))
        
        o,r,end_trial,t,env_state = environment.reinit_trial()
        
        state = init_state(params)

        rewards,observations,t_list,states,actions,true_states = [r],[o],[t],[],[],[env_state]

        
        
        while not end_trial:
            
            choice_rng_key, agent_key = jr.split(agent_key)
            state,(u_d,u_idx,u_vec) = agent_step((o,r,end_trial,t),state,params,choice_rng_key)
        
            o,r,end_trial,t,env_state = environment.step(u_vec)
            
            # The history of experiences for this trial :
            states.append(state)
            actions.append(u_vec)
            
            # And the observations and rewards for the next trial : 
            rewards.append(r)
            observations.append(o)
            t_list.append(t)
            
            # And the true state of the environment :
            true_states.append(env_state)
        
        # Last agent state update :
        choice_rng_key, agent_key = jr.split(agent_key)
        state,_ = agent_step((o,r,end_trial,t),state,params,agent_key)
        states.append(state)
        
        
        # Parameter update (once every trial)
        def _swaplist(_list):
            """ Put the various factors / modalities as the leading dimension for a 2D list of lists."""
            if _list is None :
                return None
            
            for el in _list :
                if (type(el) != list) and (type(el) != tuple):
                    # There is a single factor here ! 
                    return _list
            
            _swapped_list = []
            for factor in range(len(_list[0])):
                _swapped_list.append([_el[factor] for _el in _list])
            return _swapped_list
        
        # Let's refactor the saved data : 
        ref_observations = _swaplist(observations)
        ref_states = _swaplist(states)
            
        history = (rewards,ref_observations,ref_states,actions)
        params = agent_learn(history,params)
        
        full_history["rewards"].append(rewards)
        full_history["stimuli"].append(ref_observations)
        full_history["timestamps"].append(t_list)
        full_history["states"].append(ref_states)
        full_history["params"].append(params)
        full_history["actions"].append(actions)
        full_history["env_states"].append(true_states)
      
    return params,full_history
      


# A basic class to instantiate a POMDP generative process
# This serves as a vritual training environment for our 
# RL models and as a source of synthetic data for parameter / model retrieval tests
class TrainingEnvironment :
    """
    Simulate a simple training environment based on a POMDP.    
    """
    def __init__(self,rng_key,a,b,c,d,e,u,T):
        # Environment parameters
        self.a = a
        self.b = b
        self.c = c 
        self.d = d
        self.e = e
        self.u = u
        
        # Timing
        self.Ntimesteps = T
        self.t = 0
        self.rng_key = rng_key
        
        # Inner state and last feedback
        self.current_state = None
        self.previous_observation = None
        
        self.update_vectorized_weights()
    
    def update_vectorized_weights(self):
        self.vec_a,self.vec_b,self.vec_d = vectorize_weights(self.a,self.b,self.d,self.u)
    
    def reinit_trial(self):
        self.t = 0
        
        init_tmstp_key,self.rng_key = jax.random.split(self.rng_key)
        [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = initial_state_and_obs(init_tmstp_key,self.vec_a,self.vec_d)
        
        self.current_state = s_vect
        self.previous_observation = o_vect
        
        return o_vect,jnp.array(0.0),False,self.t,s_idx
    
    def step(self,action_chosen):
        self.t = self.t + 1   # New timestep !
        
        if self.t == self.Ntimesteps: # This should not happen as we check below :
            print("New trial ! The action has not been used here.")
            return self.reinit_trial()
        
        timestep_rngkey,self.rng_key = jax.random.split(self.rng_key)
        [s_d,s_idx,s_vect],[o_d,o_idx,o_vect] = process_update(timestep_rngkey,self.current_state,self.vec_a,self.vec_b,action_chosen)
        
        
        N_outcomes = o_vect[0].shape[0]
        reward = (jnp.linspace(0,1,N_outcomes)*(o_vect[0] - self.previous_observation[0])).sum()
        
        
        # The next timestep for the agent :
        
        self.current_state = s_vect
        self.previous_observation = o_vect
         
        return o_vect,reward,(self.t == self.Ntimesteps-1),self.t,s_idx



def generate_synthetic_data(environment_object,agent_functions,n_trials=10,seed=0):
    # Synthetic data (here, generated randomly) :
    params_final,training_hist = run_loop(environment_object,agent_functions,seed,n_trials)


    # Parameter update (once every trial)
    def _swaplist(_list):
        """ Put the various factors / modalities as the leading dimension for a 2D list of lists."""
        if _list is None :
            return None
        
        for el in _list :
            if (type(el) != list) and (type(el) != tuple):
                # There is a single factor here ! 
                return _list
        
        _swapped_list = []
        for factor in range(len(_list[0])):
            _swapped_list.append([_el[factor] for _el in _list])
        return _swapped_list
            
            
    formatted_stimuli= [jnp.array(o) for o in _swaplist(training_hist["stimuli"])]
    bool_stimuli = [jnp.ones_like(stim[...,0]) for stim in formatted_stimuli]
    rewards = jnp.array(training_hist["rewards"])
    actions = jnp.array(training_hist["actions"])
    tmtsp = jnp.array(training_hist["timestamps"])
    synthetic_data = (formatted_stimuli,bool_stimuli,rewards,actions,tmtsp)
    
    return synthetic_data