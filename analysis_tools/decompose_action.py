import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import vmap

from functools import partial

# Actions performed : this encompasses the points dropped
# But may also include temporal elements such as :
# - the time taken to perform an actions (first point / second point)
# - when the action was performed with regard to the gauge


# canvas_size = TASK_RESULTS[0][0]["canvas_size"]
# all_actions_data = np.stack([subjdata[2]["blanket"]["actions"] for subjdata in TASK_RESULTS]).astype(float)

def decompose_all_actions(action_array,canvas_size=(750,750),
                          distance_bins=jnp.array([0.0,0.1,0.3,0.6,jnp.sqrt(2) + 1e-10]),
                           angle_N_bins = 4,
                           position_N_bins_per_dim = 3):
    Nsubj,Ntrials,Nactions,Npoints,Nfeatures = action_array.shape
    
    # Npoints is always 2 for this experiment 
    
    # Nfeatures is always 4, with the following coordinates :
    #  - 0 and 1 being the x and y positions of the points (warning, y is inverted for usual Python indexing)
    #  - 2 being the time between the dropping of this point and the first one (0 for the first and up to 7000 for the second)
    #  - 3 being 1 if a point was actually dropped and 0 if it is a placeholder
    
    mapped_function = partial(decompose_one_action,canvas_size=canvas_size,
                                distance_bins= distance_bins,
                                angle_N_bins = angle_N_bins,
                                position_N_bins_per_dim = position_N_bins_per_dim)
    
    mapped_over_all = vmap(vmap(vmap(mapped_function)))
    
    
    return mapped_over_all(action_array)


def decompose_one_action(points,
                         canvas_size=(750,750),
                         distance_bins=jnp.array([0.0,0.1,0.3,0.6,jnp.sqrt(2) + 1e-10]),
                         angle_N_bins = 4,
                         position_N_bins_per_dim = 3):
    """_summary_
    Transofrm the measured values in the subject action screen to a discrete (one-hot) categorical encoding
    to use in fitting procedures. 
    A more complex model of this would involve considering that each subject set of points is sampled from a gaussian
    which parameters depend on the action chosen by the discrete planner. TODO !
    
    """
    
    
    assert points.shape == (2,4),"Points should have shape (2,4) but have shape " + str(points.shape)
    
    # Valid action array :
    valid_action = points[0,3]*points[1,3] # 1 if the action is valid, 0 otherwise
    
    
    # All of the following elements make no sense if the action is not valid !
    
    # Then, let's decompose this action properly :
    # Normalize the point data positions :
    Xnormpoints = points[...,0]/canvas_size[0]
    Ynormpoints = 1.0 - points[...,1]/canvas_size[1]

    normpoints = jnp.stack([Xnormpoints,Ynormpoints],axis=-1)
            # Npoints x Ncoordinates(2)

    # Coordinate 1 : Barycenter (x,y) ------------------------------------------------
    middle_point = (normpoints[0,:2]+normpoints[1,:2])/2.0
    # If we need to consider incomplete actions :
    # barycenter_x = (all_actions_data[...,0,0]*first_point_weights+all_actions_data[...,1,0]*second_point_weights)/(first_point_weights+second_point_weights+1e-10)
    # barycenter_y = (all_actions_data[...,0,1]*first_point_weights+all_actions_data[...,1,1]*second_point_weights)/(first_point_weights+second_point_weights+1e-10)
    # barycenters = np.stack([barycenter_x,barycenter_y],axis=-1)

    
    vectAB = normpoints[1,:]-normpoints[0,:]


    # Coordinate 2 : Distance between points (=norm of the vector) ------------------------------------------------
    norm_distance = jnp.linalg.norm(vectAB,2)
    
    # Coordinate 3 : Angle between the horizontal and the vector A-B ------------------------------------------------
    angles = jnp.atan2(vectAB[...,1],vectAB[...,0])
        
    
    
    # We got 3 scalar values, let's discretize them !
    dig_pos_idx,dig_pos_vect = discretize_position(middle_point,N_bins_per_dim=position_N_bins_per_dim)
    dig_dist_idx,dig_dist_vect = discretize_distance(norm_distance, static_bins=distance_bins)
    dig_angle_idx,dig_angle_vect,angle_bins = discretize_angle(angles,N_bin_categories = angle_N_bins)
    
    return (dig_pos_idx,dig_dist_idx,dig_angle_idx),(dig_pos_vect,dig_dist_vect,dig_angle_vect),valid_action


def discretize_angle(angle_rad,N_bin_categories = 8):
    """_summary_
    Args:
        angle_rad (scalar): an angle value, between -pi and pi
        N_bin_categories (int, optional): How many angles should we discretize. A multiple of 4. Defaults to 4.

    Returns:
        _type_: A one-hot categorical encoding of the radian angle, where the center of each bin is (k*pi/N_bin_categories)-pi
    """
    # angle_rad 
    # N_bin_categories should be a multiple of 4 (or 8 !)

    # The middle of the cardinal points should be the center of their respective bins
    half_a_bin = (2*jnp.pi/(N_bin_categories))*0.5
    bins = jnp.linspace(-np.pi + half_a_bin,np.pi - half_a_bin,N_bin_categories) 

    dig_angle_raw = jnp.digitize(angle_rad,bins)
    dig_angle = jnp.where(dig_angle_raw>N_bin_categories-1,0,dig_angle_raw) 
                    # We want N_bin_categories, so the ones that go beyond the limit are counted as "0" (around ~-pi / pi / ~180° in the circle)

    vectorized_dig_angle = jax.nn.one_hot(dig_angle,N_bin_categories)

    return dig_angle,vectorized_dig_angle,bins

def discretize_distance(norm_dist, static_bins=jnp.array([0.0,0.1,0.3,0.6,jnp.sqrt(2) + 1e-10])):
    """_summary_
    Args:
        angle_rad (scalar): an angle value, between -pi and pi
        N_bin_categories (int, optional): How many angles should we discretize. A multiple of 4. Defaults to 4.

    Returns:
        _type_: A one-hot categorical encoding of the radian angle, where the center of each bin is (k*pi/N_bin_categories)-pi
    """
    N_bin_categories = static_bins.shape[0]-1
    dig_dist = (jnp.digitize(norm_dist,static_bins)-1).astype(int)
    vectorized_dig_dist = jax.nn.one_hot(dig_dist,N_bin_categories)
    return dig_dist,vectorized_dig_dist
    

def discretize_position(norm_pos,N_bins_per_dim=3,eps = 1e-5):   
    bins = jnp.linspace(0.0-eps,1.0+eps,N_bins_per_dim+1) 
    
    # We discretize each components independently : 
    discr_x = jnp.digitize(norm_pos[0],bins)-1
    discr_y = jnp.digitize(norm_pos[1],bins)-1
    dig_position = (discr_x*N_bins_per_dim + discr_y).astype(int)  # 0 to N_bins_per_dim²-1
    vectorized_dig_position = jax.nn.one_hot(dig_position,N_bins_per_dim*N_bins_per_dim)
    # x_position = 
    return dig_position,vectorized_dig_position
    
if __name__=="__main__":
    
    A = jnp.array([1.840e+02,5.330e+02,0.000e+00,1.000e+00])
    B = jnp.array([5.840e+02,1.330e+02,4.60e+02,1.000e+00])
    print(decompose_one_action(jnp.stack([A,B])))
    
    
    # x = jnp.linspace(-jnp.pi,jnp.pi,1000)
    # y = discretize_angle(x)
    # plt.plot(x,y)
    # plt.show()
    # print(y)
    
    
    
    # x = jnp.linspace(0.0,1.0,1000)
    # y = discretize_distance(x)
    # print(y)
    # plt.plot(x,y)
    # plt.show()
    xs = jnp.linspace(0,1.0,1000)
    ys = jnp.linspace(0,1.0,1000)
    
    
    positions = jnp.stack([xs,ys],axis=-1)
    

    for x in range(50):
        for y in range(50):
            print(discretize_position(jnp.array([x/50,y/50]),N_bins_per_dim=3))