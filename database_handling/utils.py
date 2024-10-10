import numpy as np




# Utils, a priori only fit for these functions :
def get_values_in_interval(sorted_array,lower_bound=None,upper_bound=None,axis_searched=0):
    sort_axis = sorted_array[:,axis_searched]
    
    if lower_bound == None:
        lower_bound = np.min(sort_axis)
    if upper_bound == None:
        upper_bound = np.max(sort_axis)
    
    condition = (sort_axis>=lower_bound) & (sort_axis<=upper_bound)

    return sorted_array[condition]