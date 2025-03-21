import matplotlib.pyplot as plt
import numpy as np
import jax





def polar_heatmap(ax,points_data,N_bin_categories = 8,colormap = 'viridis'):
    
    assert N_bin_categories % 4 == 0, "N_bin_categories must be a multiple of 4 !"
    
    angles = np.atan2(points_data[...,1,1]-points_data[...,0,1],points_data[...,1,0]-points_data[...,0,0])
    # angles = np.reshape(angles,(Nsubj,-1))
    
    Nsubj = angles.shape[0]
    
    half_a_bin = (2*np.pi/(N_bin_categories))*0.5
    bins = np.linspace(-np.pi + half_a_bin,np.pi - half_a_bin,N_bin_categories) 

    digitized_angles = (np.digitize(angles,bins))
    digitized_angles[digitized_angles > N_bin_categories-1] = 0
    digitized_angles = digitized_angles.astype(int)


    one_hot_encoded = jax.nn.one_hot(digitized_angles,N_bin_categories,axis=-1)


    digitized_angle_counts = np.sum(np.reshape(one_hot_encoded,(Nsubj,-1,N_bin_categories)),axis=0)


    n_radius,n_angle = digitized_angle_counts.shape

    r = np.linspace(0, n_radius, n_radius)           # 100 radial points from 0 to 1
    theta = bins - half_a_bin                   # angular points from 0 to 2π
    Theta, R = np.meshgrid(theta, r)

    heatmap = ax.pcolormesh(Theta, R, digitized_angle_counts, cmap=colormap)
    ax.set_ylim([-0.2*n_radius,1*n_radius])
    return ax,heatmap
