

from database_handling.database_extract import get_all_subject_data_from_internal_task_id
from collections import defaultdict

import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpy as np
from actynf.jaxtynf.jax_toolbox import _jaxlog

def cat_kl(p,q,axis=0):
    return jnp.sum(p*(_jaxlog(p)-_jaxlog(q)),axis=axis)

def compute_js_controllability(transition_matrix):
    """ 
    An estimate of system controllability based on the Jensen Shannon Divergence learnt action transitions.
    Normalized between 0 and 1.
    """
    assert transition_matrix.ndim == 3, "JS controllability estimator expects a 3dimensionnal matrix !"
    Ns = transition_matrix.shape[0]
    
    M = jnp.mean(transitions,axis=2)  # The average transition regardless of action

    # KL dir between an action and M  : 
    kl_dirs = jnp.sum(transitions*(_jaxlog(transitions) - _jaxlog(jnp.expand_dims(M,-1))),axis=0)
    
    norm_jsd = jnp.mean(kl_dirs,axis=(0,1))/_jaxlog(Ns)

    return norm_jsd
    
# def transpose_list_of_objects():
def controll(transition_matrix):
    
    M = jnp.mean(transition_matrix,axis=(0,2))  # The average transition regardless of action
    
    r = transition_matrix*(_jaxlog(transition_matrix) - _jaxlog(jnp.expand_dims(M,(0,2))))
    
    print(jnp.mean(r,axis=(0,2)))


def mat(x,Ns):
    
    m = np.zeros((Ns,Ns,Ns))
    for u in range(Ns):
        for s_i in range(Ns):
            m[s_i,s_i,u] = 1-x
            m[u,s_i,u] = x + m[u,s_i,u]
    return m

if __name__=="__main__":   
    
    N = 100
    
    ps = np.linspace(0,1,N)
    ys = np.zeros((N,))
    
    Ns = 2
    
    for k,p in enumerate(ps) :
        # action_up = jnp.array([
        #     [1-p,0,0],
        #     [p,1-p,0],
        #     [0,p,1]
        # ])
        
        # action_down = jnp.array([
        #     [1,p,0],
        #     [0,1-p,p],
        #     [0,0,1-p]
        # ])
        
        # action_neutral = jnp.array([
        #     [1,0,0],
        #     [0,1,0],
        #     [0,0,1]
        # ])
        
        # transitions = jnp.stack([action_up,action_down],axis=-1)
        
        
        transitions = mat(p,Ns)
            
        ys[k] = compute_js_controllability(transitions)
        
    plt.plot(ps,ys)
    plt.xlabel("p_transition")
    plt.ylabel("Jensen-Shannon divergence")
    plt.title("Controllability evolution depending on learnt transition matrix")
    plt.grid()
    plt.show()