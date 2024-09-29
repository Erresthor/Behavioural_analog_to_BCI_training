import numpy as np

import jax
import jax.numpy as jnp
from jax import grad,jit,vmap
from jax.scipy.stats import norm

from functools import partial

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



if __name__=="__main__":
    std = 0.000005
    Ns = 33
    
    all_scalar_fb_values = np.linspace(0,1,Ns)   # Assume that the bigger the index of the state, the better the feedback
    
    discretize_distance_normal_function = partial(discretize_normal_pdf,std=std,num_bins = 1000,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
    stickiness,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
    
    print(all_scalar_fb_values)
    print(np.round(np.array(edges),2))
    
    
    
    print(stickiness)
    
    
    import matplotlib.pyplot as plt
    
    plt.imshow(stickiness,vmin=0,vmax=1)
    plt.show()