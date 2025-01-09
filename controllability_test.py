

from database_handling.database_extract import get_all_subject_data_from_internal_task_id
from collections import defaultdict

import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpy as np
from actynf.jaxtynf.jax_toolbox import _jaxlog

def cat_kl(p,q,axis=0):
    return jnp.sum(p*(_jaxlog(p)-_jaxlog(q)),axis=axis)

def pick_tuple_elements(A, indices):
    if isinstance(indices, (list, tuple)):
        return tuple(A[i] for i in indices)
    elif isinstance(indices, int):
        return A[indices]
    else:
        raise TypeError("indices must be a list, tuple, or int")

def re_expand(original_array_shape, array_to_re_expand, axes):
    to_do_shapes = pick_tuple_elements(original_array_shape,axes)
    if (type(axes)==int):
        axes_as_list = [axes]
    else: 
        axes_as_list = axes
    if (type(to_do_shapes)==int):
        to_do_shapes = [to_do_shapes]

    for shap_idx in range(len(to_do_shapes)):
        array_to_re_expand = np.repeat(array_to_re_expand,to_do_shapes[shap_idx],axis=axes_as_list[shap_idx])
    return array_to_re_expand

def normalize(X,axis=0,all_axes=False,epsilon=1e-15):
    if (all_axes):
        axis= tuple(range(X.ndim))                                                ###normalises a matrix of probabilities
    if(type(X) == list):
        x =[]
        for i in range(len(X)):
            x.append(normalize(X[i],axis=axis))
    elif (type(X)==np.ndarray) :
        # If this is material we can work with :
        X = X.astype(float)
        X[X<0] = 0
        # Check if normalization would result in 0 divisions: 
        # 1 : Check is sum of all elements in (axis) below threshold
        Xint =  np.sum(X,axis,keepdims=True) < epsilon
        # 2 : Make the mask the same size as the original matrix by repeating
        # along the sumed axes 
        Xint = re_expand(X.shape,Xint,axis)

        # In case of divisions by zero, we add epsilon
        X[Xint] = X[Xint] + epsilon

        # Finally, we return the actual normalized matrix
        x= X/(np.sum(X,axis,keepdims=True))
    elif(X==None):
        return None
    else :
        print("Unknwon type encountered in normalize : " + str(type(X)))
        print(X)
    return x


def compute_js_controllability(transition_matrix):
    """ 
    An estimate of system controllability based on the Jensen Shannon Divergence learnt action transitions.
    Normalized between 0 and 1.
    """
    assert transition_matrix.ndim == 3, "JS controllability estimator expects a 3dimensionnal matrix !"
    Ns = transition_matrix.shape[0]
    
    M = jnp.mean(transition_matrix,axis=2)  # The average transition regardless of action

    # KL dir between an action and M  : 
    # kl_dirs = jnp.sum(transitions*(_jaxlog(transitions) - _jaxlog(jnp.expand_dims(M,-1))),axis=0)
    kl_dirs = jnp.sum(transition_matrix*(_jaxlog(transition_matrix) - _jaxlog(jnp.expand_dims(M,-1))),axis=0)
    
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


def mat_with_prior(x,Ns):
    m = normalize(np.ones((Ns,Ns,Ns)))
    for u in range(Ns):
        m[...,u] = m[...,u]*(np.ones((Ns,Ns))-np.eye(Ns)) + x*np.eye(Ns)
    return m

if __name__=="__main__":   
    
    N = 5
    
    ps = np.linspace(-1,1,N)
    
    
    Ns = 3
    
    fig,axs = plt.subplots(Ns,3,figsize=(15,5*Ns))
    axcounter = 0
    
    plot_these = [0,int(N/2),N-1]
    
    ys = np.zeros((N,))
    for k,p in enumerate(ps) :
        transitions = mat_with_prior(p,Ns)
        ys[k] = compute_js_controllability(transitions)
        
        if k in plot_these:
            axs[0,axcounter].set_title("p = {:.2f}".format(p))
            for s in range(Ns):
                axs[s,axcounter].imshow(transitions[...,s],vmin=-1,vmax=1)
            axcounter = axcounter + 1
    
    
    fig,ax = plt.subplots()
    ax.plot(ps,ys)
    for k in plot_these:
        ax.axvline(ps[k],color="red")
    ax.set_xlabel("p_transition")
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Controllability evolution depending on learnt transition matrix")
    ax.grid()
    plt.show()