import numpy as np

import jax
import jax.numpy as jnp
from jax import grad,jit
from jax.scipy.stats import norm

from functools import partial

def remove_by_indices(iter, idxs):
    return [e for i, e in enumerate(iter) if i not in idxs]

def to_shape(a, shape,fill_with_value=0):
    x_, = shape
    x, = a.shape
    x_pad = (x_-x)
    return np.pad(a,(x_pad//2, x_pad//2 + x_pad%2),
                  mode = 'constant',
                  constant_values=fill_with_value)
    
    
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


from jax import vmap

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
    