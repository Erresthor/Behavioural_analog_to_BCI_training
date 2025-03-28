import numpy as np

import jax
import jax.numpy as jnp
from jax import grad,jit,vmap
from jax.scipy.stats import norm

from functools import partial
from actynf.jaxtynf.jax_toolbox import _jaxlog

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

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


def distance(tuple1,tuple2,normed=False,grid_size=2):
    linear_dist =  np.sqrt((tuple1[0]-tuple2[0])*(tuple1[0]-tuple2[0])+(tuple1[1]-tuple2[1])*(tuple1[1]-tuple2[1]))
    if normed :
        assert grid_size>1,"Grid should be bigger"
        gs = grid_size-1
        return linear_dist/np.sqrt(gs*gs+gs*gs)
    return linear_dist

def discretized_distribution_from_value(x,number_of_ticks):
    assert number_of_ticks>1,"There should be at least 2 different distribution values"
    return_distribution = np.zeros((number_of_ticks,))
    if (x<0.0):
        return_distribution[0] = 1.0 
    elif (x>=1.0):
        return_distribution[-1] = 1.0
    else :
        sx = x*(number_of_ticks-1)
        int_sx = int(sx)  # The lower index
        float_sx = sx-int_sx  # How far this lower index is from the true value
        return_distribution[int_sx] = 1.0-float_sx  # The closer to the true value, the higher the density
        return_distribution[int_sx+1] = float_sx
    return return_distribution

   
@partial(jit, static_argnames=["num_bins"])
def discretize_normal_pdf(mean, std, num_bins, lower_bound, upper_bound):
    """ Thank you ChatGPT ! """    
    
    # I didn't want to do this, but this may be needed to avoid unwated edge-cases : 
    # Prevent std from falling below a certain value, else the distribution is completely flat
    # which is the opposite of a very well defined observation modality: 
    K = 0.1   # Empirical constant
    std = jnp.clip(std,K*1/num_bins)
    
    # Define the bin edges
    bin_edges = jnp.linspace(lower_bound, upper_bound, num_bins + 1)
    
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Calculate PDF values at bin centers
    pdf_values = norm.pdf(bin_centers, loc=mean, scale=std)
    
    # Normalize the PDF values to sum to 1 (so it acts like a discrete distribution)
    pdf_values_normalized = pdf_values / (jnp.sum(pdf_values))
    
    return pdf_values_normalized, bin_edges   


def mat_sub2ind(array_shape, sub_tuple):
    rows, cols = sub_tuple[0],sub_tuple[1]
    if ((rows < 0)or(rows>=array_shape[0])) or ((cols < 0)or(cols>=array_shape[1])) :
        raise ValueError(str(sub_tuple) + " is outside the range for array shape " + str(array_shape))
    return cols*array_shape[0] + rows

def mat_ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


# Python versions of table indexings
def sub2ind(array_shape, sub_tuple):
    """ For integers only !"""
    rows, cols = sub_tuple[0],sub_tuple[1]
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    """ For integers only !"""
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return rows, cols


def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out



# Plotting helpers
def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')

def clever_running_mean(arr, N):
    xarr = np.array(arr)
    xpost = np.zeros(xarr.shape)
    # raw_conv = np.convolve(x, np.ones(N)/N, mode='same')
    for k in range(xarr.shape[0]):
        localmean = 0.0
        cnt = 0.0
        for i in range(k-N,k+N+1,1):
            if ((i>=0) and (i<xarr.shape[0])):
                localmean += xarr[i]
                cnt += 1
        xpost[k] = localmean/(cnt+1e-18)
    return xpost

def generate_random_vector(N,rng):
    return np.array([rng.random() for k in range(N)])

def generate_random_array(N):
    return np.random.random((N,))

def weighted_padded_roll(matrix,generalize_fadeout,roll_axes=[-1,-2]):
    assert matrix.ndim == 2,"Weighted Padded Roll only implemented for 2D arrays"
    K = matrix.shape[0]
    roll_limit = K
        
    padded_matrix = jnp.pad(matrix,((K,K),(K,K)),mode="constant",constant_values=0)
     
    rolling_func = lambda k : jnp.roll(padded_matrix,k,roll_axes)*generalize_fadeout(jnp.abs(k))
    
    all_rolled = vmap(rolling_func)(jnp.arange(-roll_limit,roll_limit+1))
    
    # Remove padding : 
    all_rolled = all_rolled[...,K:-K,K:-K]
    
    new_db = all_rolled.sum(axis=-3)
    
    return new_db



if __name__=="__main__":
    
    from actynf.jaxtynf.jax_toolbox import _normalize
    
    Ns = 5
    Nu = 9
    
    B,_ = _normalize(np.ones((Ns,Ns,Nu)))  # Transition table (tracking controlability)
    Q = np.zeros((Nu,Ns))                  # Reward table (tracking reward for each state)

    q_alpha = 0.4
    transition_alpha = 0.5
    
    
    last_reward = 0.3
    
    previous_posterior = np.array([0.1,0.8,0.1,0.0,0.0])
    new_posterior = np.array([0.0,0.1,0.8,0.1,0.0])
    
    action_selected = np.zeros((Nu,))
    action_id = 2
    action_selected[action_id] = 1.0
    
    
    previous_action_state = jnp.einsum("u,j->uj",action_selected,previous_posterior)
    observed_transition = jnp.einsum("i,j,u->iju",new_posterior,previous_posterior,action_selected) # This density should be pushed towards 1 !
    unobserved_transition = jnp.einsum("i,j,u->iju",1.0 - new_posterior,previous_posterior,action_selected) # This density should be pushed towards 0 !
    new_B = B - transition_alpha*unobserved_transition*B + transition_alpha*observed_transition*(1.0-B)
    
    
    
    gamma = 20.0
    
    # Qtable :
    previous_posterior = np.array([0.1,0.8,0.1,0.0,0.0])
    raw_action_state = jnp.einsum("u,j->uj",action_selected,previous_posterior)
    generalized_action_state = weighted_padded_roll(raw_action_state,lambda x : jnp.exp(-gamma*x),[-1])
    raw_Q = Q + q_alpha*(last_reward - Q)*raw_action_state
    gen_Q = Q + q_alpha*(last_reward - Q)*generalized_action_state
    
    # Transition table : 
    fadeout_func = lambda x : jnp.exp(-gamma*x)
    generalize_function = lambda x : weighted_padded_roll(x,fadeout_func,[-1,-2])
    
    
    raw_B = B
    gen_B = B
    
    for t in range(10):
        observed_transition = jnp.einsum("i,j->ij",new_posterior,previous_posterior)
        unobserved_transition = jnp.einsum("i,j->ij",1.0-new_posterior,previous_posterior)
        
        # Raw learnt transitions : 
        raw_db = jnp.einsum("ij,u->iju",observed_transition,action_selected)
        raw_dbo = jnp.einsum("ij,u->iju",unobserved_transition,action_selected)
        raw_B = raw_B - transition_alpha*raw_dbo*raw_B + transition_alpha*raw_db*(1.0-raw_B)
        
        gen_db = jnp.einsum("ij,u->iju",generalize_function(observed_transition),action_selected)
        gen_dbo = jnp.einsum("ij,u->iju",generalize_function(unobserved_transition),action_selected)
        gen_B = gen_B - transition_alpha*gen_dbo*gen_B + transition_alpha*gen_db*(1.0-gen_B)
        
    
        print(np.sum(raw_B[...,action_id],axis=0))
        print(np.sum(gen_B[...,action_id],axis=0))
        
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,3)
        axes[0,0].set_title("Raw obs. transitions")
        axes[0,0].imshow(raw_db[...,action_id],vmin=0,vmax=1.0)
        axes[0,1].set_title("Raw unobs. transitions")
        axes[0,1].imshow(raw_dbo[...,action_id],vmin=0,vmax=1.0)
        axes[0,2].set_title("Updated transition mapping")
        axes[0,2].imshow(raw_B[...,action_id],vmin=0,vmax=1.0)
        axes[1,0].set_title("Gen. obs. transitions")
        axes[1,0].imshow(gen_db[...,action_id],vmin=0,vmax=1.0)
        axes[1,1].set_title("Raw unobs. transitions")
        axes[1,1].imshow(gen_B[...,action_id],vmin=0,vmax=1.0)
        axes[1,2].set_title("Updated gen. transition mapping")
        axes[1,2].imshow(gen_B[...,action_id],vmin=0,vmax=1.0)
    plt.show()