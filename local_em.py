import random as ra
import numpy as np

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jax import lax,vmap, jit

from functools import partial

from actynf.jaxtynf.jax_toolbox import _normalize,_jaxlog,_condition_on
from actynf.jaxtynf.jax_toolbox import weighted_padded_roll

from actynf.jaxtynf.shape_tools import vectorize_weights,to_source_space
from actynf.jaxtynf.jax_toolbox import zero_like_tree

from generalize_helper import get_extrapolated_deltab

# For 2D observation matrices (flattened state space)
def get_log_likelihood_one_observation(o_m,a_m,obs_m_filter):    
    return _jaxlog(jnp.einsum("ij,i->j",a_m,o_m))*obs_m_filter

def get_log_likelihood_one_timestep(o,a,obs_filter):    
    return tree_map(get_log_likelihood_one_observation,o,a,obs_filter)

def get_log_likelihood_all_timesteps(o,a,observed_value_filters):
    result = vmap(get_log_likelihood_one_timestep,in_axes=(0,None,0))(o,a,observed_value_filters)
    ll = jnp.sum(jnp.stack(result), 0)    
    return ll

# E-step : given parameters, what is the most likely sequence of states ?
def forwards_pass(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d):
    r"""Forwards filtering for a history of emissions and actions defined across a single trial.
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        marginal log likelihood and filtered posterior distribution ; tensor of shape (Ntimesteps x Nstates)

    """        
    # Compute the log likelihooods of each emission (if they were observed)
    logliks = get_log_likelihood_all_timesteps(hist_obs_vect,vec_a,hist_obs_bool)
    num_timesteps, num_states = logliks.shape
    
    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        forward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        # Predict the next state (going forward in time).
        transition_matrix = jnp.einsum("iju,u->ij",vec_b,hist_u_vect[t-1,...])
                # If hist_u_vect was not observed, it may be replaced by a flat mapping ? subjects habits ? Infered ?
                # See VFE minimization with flexible transitions :) : actynf.jax_methods.utils.misc_vfe_minimization_action_mdp.py

        prior_next_timestep = transition_matrix @ forward_filt_probs
                
        return (carry_log_norm,prior_next_timestep),forward_filt_probs

    init_carry = (jnp.array([0.0]),vec_d)
    (log_norm,_),forward_pred_probs = lax.scan(_step, init_carry, jnp.arange(num_timesteps))
    return log_norm,forward_pred_probs

def backwards_pass(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b):
    r"""Run the filter backwards in time. This is the second step of the forward-backward algorithm.

    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        filtered posterior distribution ; tensor of shape (Ntimesteps x Nstates)
    Returns:
        marginal log likelihood and backward messages ; tensor of shape (Ntimesteps x Nstates)

    """     
    # Compute the log likelihooods of each emission (if they were observed)
    logliks = get_log_likelihood_all_timesteps(hist_obs_vect,vec_a,hist_obs_bool)
    
    num_timesteps, num_states = logliks.shape

    def _step(carry, t):
        carry_log_norm,prior_this_timestep = carry
        
        backward_filt_probs, log_norm = _condition_on(prior_this_timestep, logliks[t])
        
        carry_log_norm = carry_log_norm + log_norm
        
        # Predict the next (previous) state (going backward in time).
        transition_matrix = jnp.einsum("iju,u->ij",vec_b,hist_u_vect[t-1,...])
        prior_previous_timestep = transition_matrix.T @ backward_filt_probs
                
        return (carry_log_norm,prior_previous_timestep),backward_filt_probs

    init_carry = (jnp.array([0.0]),jnp.ones(num_states))
    (log_norm,_),backward_pred_probs_rev = lax.scan(_step, init_carry, jnp.arange(num_timesteps)[::-1])
    
    backward_pred_probs = backward_pred_probs_rev[::-1]
    return log_norm, backward_pred_probs

def smooth_trial(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d):
    r"""Forwards-backwards filtering for a history of emissions and actions defined across a single trial.
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape ((Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        smoothed posterior distribution given the proposed parameters ; tensor of shape (Ntimesteps x Nstates)
    """    
    ll_for,forwards_smooths = forwards_pass(
                hist_obs_vect,hist_u_vect,hist_obs_bool,
                vec_a,vec_b,vec_d
    )
    ll_back,backwards_smooths = backwards_pass(
            hist_obs_vect,hist_u_vect,hist_obs_bool,
            vec_a,vec_b
    )
    smoothed_posterior,_ = _normalize(forwards_smooths*backwards_smooths,axis=-1)
    return smoothed_posterior,ll_for

def smooth_trial_window(hist_obs_vect,hist_u_vect,
                  hist_obs_bool,
                  vec_a,vec_b,vec_d):
    r"""Forwards-backwards filtering for a history of emissions and actions defined across a window of several trials (leading dimension).
    Args:
        Data :
        - hist_obs_vect is a list of tensors (one tensor per modality). Each tensor is of shape (Ntrials x Ntimesteps x Noutcomes(modality))
        - hist_obs_bool is a list of tensors (one tensor per modality). Each tensor is of shape (Ntrials x Ntimesteps) and has value
            1.0 if the value was observed and 0.0 if it was not.
        - hist_u_vect is a single tensor of shape (Ntrials x (Ntimesteps-1) x Nactions)
        Parameters :
        - vec_d: flattened initial state $p(z_1 \mid \theta)$ ; tensor of shape (Nstates)
        - vec_b: flattened transition matrix $p(z_{t+1} \mid z_t, u_t, \theta)$ ; tensor of shape (Nstates x Nstates x Nactions)
        - vec_a: flattened list of emission mappings across modalities $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$  ; list of tensors of shape (Noutcomes(modality) x Nstates)
        
    Returns:
        smoothed posterior distribution given the proposed parameters ; tensor of shape (Ntrials x Ntimesteps x Nstates)
    """  
    return vmap(smooth_trial,in_axes=(0,0,0,None,None,None))(hist_obs_vect,hist_u_vect,hist_obs_bool,vec_a,vec_b,vec_d)

# M-step : given the state posterior, find the best matching parameters : 
# This is equivalent to the learning phase of Active Inference : 

# _______________________________________________________________________________________
# Playing with allowable actions : 
# switching from a vectorized space (all actions in one dimension)
# to a factorized space (allowable actions for each factor)
def vectorize_factorwise_allowable_actions(_u,_Nactions):
    """ 
    From an array of indices of actions
     - /!\,_u should be 2 dimensionnal (Nfactors x Nactions)
    To a list of arrays of size (N_allowable_actions x Npossible_transitions_for_factor)
    """
    
    assert _u.ndim == 2,"_u should be a 2 dimensionnal mapping between action and transitions but is a {} dimensional tensor".format(_u.ndim)
    
    # This needs to be mapped across state factors ! 
    def factorwise_allowable_action_vectors(idx,N_actions_f):
        return jax.nn.one_hot(idx,N_actions_f)
    
    # This function takes one of the action index, and decomposes it into action vectors across all factors
    map_function = (lambda _x : tree_map(factorwise_allowable_action_vectors,list(_x),_Nactions))
    
    return (vmap(map_function)(_u))
    
def posterior_transition_index_factor(transition_list,history_of_actions):
    def posterior_transition_factor(allowable_action_factor):
        return jnp.einsum("ij,kti->ktj",allowable_action_factor,history_of_actions)
    return tree_map(posterior_transition_factor,transition_list)
# _______________________________________________________________________________________

# _______________________________________________________________________________________
# ________________________________ Dirichlet Weights Updating ___________________________
# _______________________________________________________________________________________

# Parameter updating terms given a hsitory of state inferences and observed values

# emissions :
def get_delta_a(hist_obs,hist_obs_bool,
                hist_qs,
                hidden_state_shape):
    def _delta_a_mod(o_mod,o_filter_mod):
        return jnp.reshape(jnp.einsum("ti,t,tj->ijt",o_mod,o_filter_mod,hist_qs).sum(axis=-1),(-1,)+hidden_state_shape) 
    return tree_map(_delta_a_mod,hist_obs,hist_obs_bool)

# transitions :
def get_delta_b(hist_u_tree,hist_u_filter,hist_qs_tree,state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    CLIP_EXTRAPOLATED_ACTIONS = False
    if (type(state_generalize_function)==list):
        assert len(state_generalize_function)==len(hist_u_tree),"If the generalizer is a list, its length should match the number of state factors"
        
        function_to_map = partial(get_extrapolated_deltab,action_filter=hist_u_filter,generalize_action_table=action_generalize_table,cross_action_extrapolation_coeff=cross_action_extrapolation_coeff,option_clip=CLIP_EXTRAPOLATED_ACTIONS)
        return tree_map(function_to_map,hist_u_tree,hist_qs_tree,state_generalize_function)
    
    else :
        function_to_map = partial(get_extrapolated_deltab,action_filter=hist_u_filter,generalize_state_function=state_generalize_function,generalize_action_table=action_generalize_table,cross_action_extrapolation_coeff=cross_action_extrapolation_coeff,option_clip=CLIP_EXTRAPOLATED_ACTIONS)
        return tree_map(function_to_map,hist_u_tree,hist_qs_tree)

# initial states :
def get_delta_d(hist_qs_tree):
    def _delta_d_factor(hist_qs_factor):
        return hist_qs_factor[0,:]
    
    return tree_map(_delta_d_factor,hist_qs_tree)


# Compute parameter update terms depending on options.
# This is operated at the trial level !
# meant to be vectorized along the trial dimension for the first 3 arguments.
def get_parameter_update(hist_obs_vect,hist_factor_action_vect,
                         hist_obs_bool,hist_factor_action_bool,
            smoothed_posteriors,
            Ns,Nu,
            state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    r"""
    - Ns is the hidden state space shape.
    """    
    # For the state posteriors, factorize them after smoothing !
    factorized_smoothed_posteriors = vmap(lambda x : to_source_space(x,Ns))(smoothed_posteriors)
    
    # Warning ! When we missed observations or actions, we can't use it to 
    # update our parameters ! 
    
    # learning a is done in vectorized mode
    delta_a = get_delta_a(hist_obs_vect,hist_obs_bool,smoothed_posteriors,Ns)       
    
    delta_d = get_delta_d(factorized_smoothed_posteriors)
        
    delta_b = get_delta_b(hist_factor_action_vect,hist_factor_action_bool,factorized_smoothed_posteriors,
                          state_generalize_function,action_generalize_table,cross_action_extrapolation_coeff)
    
    
    # c and e are not implemented yet 
    # (but should probably be guided by hierarchical processes anyways)... 
    delta_c = tree_map(lambda x : jnp.zeros((x.shape[-1])),hist_obs_vect)
    
    delta_e = jnp.zeros((Nu,))
        
    return delta_a,delta_b,delta_c,delta_d,delta_e


# Main issue with this algorithm : compress multiple state factors. 
# Sophisticated inference chooses to compute belief propagation using a single latent dimension.
# Thus, we use a kronecker product to "flatten" multiple state dimensions, when they exist

# This is a problem when comes the time to learn transition matrix, especially when the model is equipped
# with some generalization ability along specific factors.
# We need to flatten the B and D matrices for the E-step, 
# and then marginalize the resulting state posteriors for the M-step


@partial(jit,static_argnames=["N_iterations","is_learn_a","is_learn_b","is_learn_d","state_generalize_function"])
def EM_jax(vec_emission_hist,emission_bool_hist,
           vec_action_hist,action_bool_hist,
           true_a_prior,true_b_prior,true_d_prior,U,
           N_iterations = 16,
           lr_a=1.0,lr_b=1.0,lr_d=1.0 ,
           is_learn_a = True,is_learn_b = True,is_learn_d = True,
           state_generalize_function=None,action_generalize_table=None,cross_action_extrapolation_coeff=0.1):
    # Checks :
    for mod in range(len(vec_emission_hist)):
        assert vec_emission_hist[mod].ndim==3, "Observations should have 3 dimensions : Ntrials x Ntimesteps x Noutcomes but has " + str(vec_emission_hist[mod].ndim) + " for modality " + str(mod)
    assert vec_action_hist.ndim==3, "Observed actions should have 3 dimensions : Ntrials x (Ntimesteps-1) x Nu"
    assert action_bool_hist.shape == vec_action_hist.shape[:-1], "The action filter should be of shape Ntrials x (Ntimesteps-1)"
    
    assert U.ndim == 2,"U should be a 2 dimensionnal mapping between action and transitions but is a {} dimensional tensor".format(U.ndim)
    
    # Static shapes
    Nmod = len(true_a_prior)
    Nf = len(true_b_prior)
    Nu,Nf2 = U.shape                                # How many allowable actions there are
                                                    # Each allowable action results in a specific transition for each factor
    
    assert Nf2==Nf,"Mismatch in the number of state factors. Please check the function inputs."
                                       
                                                    
    Nuf = [b_f.shape[-1] for b_f in true_b_prior]   # How many transitions are possible per factor
    
    Ns = true_a_prior[0].shape[1:] # This is the shape of the hidden state space (fixed for this model)
    
    # Normalize the options to match the factors / modalities
    # (a.k.a all parameters should be lists of objects!)
    def _norm_option(_field,_list_shape):
        if type(_field) != list:
            _field_value = _field # Unecessary step ?
            _field = [_field_value for k in range(_list_shape)]
        assert len(_field)==_list_shape, "Length of field " + str(_field) + " does not match the required dimension " + str(_list_shape)
        return _field
    lr_a, is_learn_a = _norm_option(lr_a,Nmod),_norm_option(is_learn_a,Nmod)
    lr_b, is_learn_b = _norm_option(lr_b,Nf),_norm_option(is_learn_b,Nf)
    lr_d, is_learn_d = _norm_option(lr_d,Nf),_norm_option(is_learn_d,Nf)
    
    # Let's get a factorized version of the action history : 
    vec_transition_per_factor = vectorize_factorwise_allowable_actions(U,Nuf)
    vec_transition_history = posterior_transition_index_factor(vec_transition_per_factor,vec_action_hist)
                        # A list of transitions performed per factor !
                        # across Ntrials x (Ntimesteps-1)
                        # y axis : states(t+1), x axis : states (t)
    
    # Useful functions :  
    get_param_variations = partial(get_parameter_update,Ns=Ns,Nu = Nu,state_generalize_function = state_generalize_function,
                                   action_generalize_table=action_generalize_table,cross_action_extrapolation_coeff=cross_action_extrapolation_coeff)
    
    def _update_prior(_prior,_param_variation,_lr=1.0,_learn_bool=True):
        if _learn_bool:
            # Here, we sum across all past trials. We could implement a
            # version where further trials impact parameter updates less !
            # (some kind of memory loss)
            return _prior + _lr*_param_variation.sum(axis=0)
        return _prior
    
    # The actual EM : for N iterations / until convergence 
    # we will alternate hidden state estimation
    # and parameter updates :
    def _scanner(carry,xs):
        # These are the parameters for this e-step iteration
        _it_a, _it_b,_it_d = carry
        
        _it_vec_a,_it_vec_b,_it_vec_d = vectorize_weights(_it_a,_it_b,_it_d,U)
        smoothed_posteriors,ll_trials_it = smooth_trial_window(vec_emission_hist,vec_action_hist,emission_bool_hist,
                                                _it_vec_a,_it_vec_b,_it_vec_d)
            # A Ntrials x Ntimesteps x Ns tensor of smoothed state posteriors !
        
        # TODO : add log prob w.r.t. parameter priors to this !
        # see dynamax fit_em : 
        # lp = self.log_prior(params) + lls.sum()
        # Where for categorical distributions, the conjugate prior is the dirichlet distribution :
        # def log_prior(self, params):
        #      return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params.probs).sum()
        # This should be quite similar to how we compute the EFE with learnable dynamics !
        delta_a,delta_b,_,delta_d,_ = vmap(get_param_variations)(vec_emission_hist,vec_transition_history,
                                                                 emission_bool_hist,action_bool_hist,
                                                                 smoothed_posteriors)
        
        _new_a = tree_map(_update_prior,true_a_prior,delta_a,lr_a,is_learn_a)
        _new_b = tree_map(_update_prior,true_b_prior,delta_b,lr_b,is_learn_b)
        _new_d = tree_map(_update_prior,true_d_prior,delta_d,lr_d,is_learn_d)
        
        return (_new_a,_new_b,_new_d),(smoothed_posteriors,ll_trials_it)

    init_carry = (true_a_prior,true_b_prior,true_d_prior)
    (final_a,final_b,final_d),(smoothed_states,lls) = lax.scan(_scanner, init_carry, jnp.arange(N_iterations))
    
    # Last smoothed posterior : 
    vec_final_a,vec_final_b,vec_final_d = vectorize_weights(final_a,final_b,final_d,U)
    final_smoothed_posteriors,final_ll = smooth_trial_window(vec_emission_hist,vec_action_hist,emission_bool_hist,
                                            vec_final_a,vec_final_b,vec_final_d)
    
    return (final_a,final_b,final_d,vec_final_d),final_smoothed_posteriors,jnp.concatenate([lls,jnp.expand_dims(final_ll,-3)],axis=-3)



if __name__=="__main__":
    Ns_all = [2,3]
    Ntrials = 2
    Ntimesteps = 3
    
    # Transitions : 2 factors
    transitions = [np.zeros((Ns,Ns,Ns)) for Ns in Ns_all]
    for f,b_f in enumerate(transitions):
        for action in range(b_f.shape[-1]):
            
            b_f[...,action] = np.eye(Ns_all[f])
            try :
                b_f[action+1,action,action] += 1.0
            except :
                b_f[0,action,action] += 1.0
    raw_b = [jnp.array(b_f) for b_f in transitions]
    
    
    raw_d = [jnp.array([0.5,0.5]),jnp.array([0.5,0.5,0.0])]
    
    
    
    raw_a = [np.zeros((2,2,3)),np.zeros((3,2,3))]

    for s in range(3):
        raw_a[0][:,:,s] = np.array([
            [0.8,0.3],
            [0.2,0.7]
        ])
        raw_a[1][:,:,s] = ([
            [1.0,0.0],
            [0.0,1.0],
            [1.0,1.0]
        ])
    
    
    u = jnp.array([
        [0,0],
        [0,1],
        [1,2]
    ])
    
    vec_a,vec_b,vec_d = vectorize_weights(raw_a,raw_b,raw_d,u)
    
    
    obs_u = np.zeros((Ntrials,Ntimesteps-1,3))
    obs_u[0,...] = np.array([
        [1,0,0],
        [0,1,0]
    ])
    obs_u[1,...] = np.array([
        [1,0,0],
        [1,0,0]
    ])
    obs_u = jnp.array(obs_u)
    obs_u_filter = jnp.ones_like(obs_u[...,0])
    
    Nuf = [b_f.shape[-1] for b_f in raw_b] 
    vec_transition_per_factor = vectorize_factorwise_allowable_actions(u,Nuf)
    vec_transition_history = posterior_transition_index_factor(vec_transition_per_factor,obs_u)

    filters = [jnp.array([[1.0,1.0,1.0],[1.0,1.0,1.0]]),
               jnp.array([[1.0,0.0,0.0],[1.0,1.0,1.0]])]
    
    o_d_1 = [jnp.array([
        [0.9,0.1],
        [0.8,0.2],
        [0.0,1.0]
    ]),jnp.array([
        [0.9,0.1,0.0],
        [1.0,1.0,1.0],
        [1.0,1.0,1.0]
    ])]
    o_d_2 = [jnp.array([
        [0.9,0.1],
        [0.2,0.8],
        [0.0,1.0]
    ]),jnp.array([
        [0.9,0.1,0.0],
        [1.0,0.0,0.0],
        [1.0,0.0,0.0]
    ])]
    obs = [jnp.stack([o1,o2],axis=0) for o1,o2 in zip(o_d_1,o_d_2)]

    
    # smoothd_post = smooth_trial_window(obs,obs_u,
    #               filters,
    #               vec_a,vec_b,vec_d)
    # print(np.round(np.array(smoothd_post),2))
    
    # param_updater = partial(get_parameter_update,Ns=tuple(Ns_all),Nu = u.shape[0],generalize_fadeout_function = None)
    # da,db,dc,dd,de = vmap(param_updater)(obs,vec_transition_history,smoothd_post)
    # print(da[0].shape)
    # print(da[1].shape)
    alpha = 1.0
    
    transition_generalizer =None# (lambda x : jnp.exp(-alpha*x))
    
    print(obs_u.shape)
    (final_a,final_b,final_d),final_post,final_ll = EM_jax(obs,filters,obs_u,obs_u_filter,
           raw_a,raw_b,raw_d,u,
           lr_a=1.0,lr_b=1.0,lr_d=1.0 ,
           is_learn_a = True,is_learn_b = True,is_learn_d = True,
           transition_generalizer=transition_generalizer)
    
    print(final_a)
    print(np.round(np.array(final_post),2))
    print(final_post.shape)
    print(final_ll[...,0])
    exit()

