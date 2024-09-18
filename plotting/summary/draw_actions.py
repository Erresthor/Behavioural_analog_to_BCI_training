import pymongo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from itertools import chain



def draw_actions(actions_array,
                 canvas_x,canvas_y,
                 start_color = np.array([0.0,0.0,255]),end_color = np.array([255,0.0,0.0])):
    if type(actions_array)==np.ndarray:
        Ntrials,Ntimesteps,Npoints,Ndata = actions_array.shape
        
        fig,axes = plt.subplots(Ntrials,Ntimesteps)

        for trial_i in range(Ntrials):
            ax_line = axes[trial_i,:]
            for timestep_j in range(Ntimesteps):
                ax = ax_line[timestep_j]
                img = 255*np.ones((canvas_x,canvas_y,3), dtype=np.uint8)
                
                
                timestep_points = actions_array[trial_i,timestep_j,...]
                cnt= 0
                point_n = Npoints
                for point in timestep_points:
                    ratio = (0 if (point_n<=1) else (cnt/(point_n-1)))
                    this_point_color = ((start_color+ratio*(end_color-start_color))/255)#.astype(np.uint8)

                    rad = 5

                    ax.plot(point[0], point[1], marker="o", markersize=rad,color=this_point_color)
                    cnt+=1
                
                point0 = timestep_points[0]
                point1 = timestep_points[1]
                ax.arrow(point0[0], point0[1], point1[0]-point0[0], point1[1]-point0[1])
                
                ax.set_title(str(timestep_j))
                ax.set_xlim([0,canvas_x])
                ax.set_ylim([0,canvas_y])
                ax.invert_yaxis()

            for ax in ax_line:
                ratio = 1.0
                x_left, x_right = ax.get_xlim()
                y_low, y_high = ax.get_ylim()
                ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
        return fig
