import numpy as np

import jax
import jax.numpy as jnp
from jax import grad,jit,vmap
from jax.scipy.stats import norm

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map

from functools import partial
from actynf.jaxtynf.jax_toolbox import _jaxlog,_normalize


def uniform_sample_leaf(_rng_leaf,_range_leaf, size):
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
    if _range_leaf.shape[0] ==3 :
        return jnp.squeeze(jr.uniform(_rng_leaf,(size,_range_leaf[-1]),minval  = _range_leaf[0], maxval =_range_leaf[1]))
    
    return jr.uniform(_rng_leaf,(size,),minval  = _range_leaf[0], maxval =_range_leaf[1])

def sample_dict_of_categoricals(dict_of_probs,rng_key):
    samples,vect_samples = {},{}
    for action_dim, probs in dict_of_probs.items():
        # Split key for each sampling operation
        rng_key, subkey = jr.split(rng_key)
        
        # Sample from the categorical distribution using each probability vector
        sample = jr.categorical(subkey, _jaxlog(probs))
        
        samples[action_dim] = sample
        vect_samples[action_dim] = jax.nn.one_hot(sample,probs.shape[0])
    
    return samples,vect_samples

def compute_js_controllability(transition_matrix):
    """ 
    An estimate of system controllability based on the Jensen Shannon Divergence learnt action transitions.
    Normalized between 0 and 1.
    """
    if type(transition_matrix)==list:
        
        # Flatten the tree to extract all the scalar values
        leaves, _ = jax.tree_util.tree_flatten(tree_map(compute_js_controllability,transition_matrix))

        # Convert the list of leaves to a JAX array and compute the mean
        mean_value = jnp.mean(jnp.array(leaves))

        return mean_value
    
    assert transition_matrix.ndim == 3, "JS controllability estimator expects a 3dimensionnal matrix !"
    Ns = transition_matrix.shape[0]
    
    M,_ = _normalize(jnp.mean(transition_matrix,axis=2))  # The average transition regardless of action

    # KL dir between an action and M  : 
    # kl_dirs = jnp.sum(transitions*(_jaxlog(transitions) - _jaxlog(jnp.expand_dims(M,-1))),axis=0)
    kl_dirs = jnp.sum(transition_matrix*(_jaxlog(transition_matrix) - _jaxlog(jnp.expand_dims(M,-1))),axis=0)/_jaxlog(Ns)
    
    norm_jsd = jnp.mean(kl_dirs,axis=(0,1))
    
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