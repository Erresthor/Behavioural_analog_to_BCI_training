import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline
from .models_utils import ind2sub

def plot_grid(ax,grid_shape,goal_state,color=np.array([0.0,0.0,0.0,0.5])):
    lw = 20*1e-3
    # ax.set_xlim(-0.5-lw,grid_shape[0]-0.5+lw)
    # ax.set_ylim(-0.5-lw,grid_shape[1]-0.5+lw)
    
    ax.set_axis_off()
    ax.set_aspect('equal', 'box')
    for _x in np.arange(grid_shape[0]+1):
        ax.axvline(_x-0.5,color=color)
    
    for _y in np.arange(grid_shape[1]+1):
        ax.axhline(_y-0.5,color=color)
        
    # Draw a little flag in the goal state !
    COLOR = "green"#np.array([0.3,0.8,0.3,1.0])
    x_goal,y_goal = goal_state
    # The pole    
    ax.plot([x_goal-0.1,x_goal-0.1],[y_goal-0.4,  y_goal+0.3],color=COLOR,marker="o", linestyle='-')
    # The flag
    ax.plot([x_goal-0.1,x_goal + 0.4],[y_goal-0.1,y_goal+(0.3-0.1)/2.0],color=COLOR,lw = 2)       
    ax.plot([x_goal-0.1,x_goal + 0.4],[y_goal+0.3,y_goal+(0.3-0.1)/2.0],color=COLOR,lw = 2)  


def plot_trajectory(ax,state_idxs_trial,
                    grid_shape,
                    color=np.array([1.0,0.0,0.0,0.5]),interp_spline_param = 2,
                    lw = 4,alpha=0.2):
    
    xs,ys = ind2sub((grid_shape),state_idxs_trial)
    
    sigma = 0.1
    xs = xs + np.random.normal(0.0,sigma,xs.shape)
    ys = ys + np.random.normal(0.0,sigma, ys.shape)

    # Plotting the interpolated trajectory
    
    param = np.linspace(0, 1, xs.size)
    spl = make_interp_spline(param, np.c_[xs,ys], k=interp_spline_param) #(1)
    
    xnew, y_smooth = spl(np.linspace(0, 1, xs.size * 100)).T #(2)
    
    
    ax.plot(xnew, y_smooth,color=color,linewidth = lw,alpha=alpha)
    
    
    ax.scatter(xs[0],ys[0],s = 200,marker="+", color = color,alpha=alpha)
    ax.scatter(xs[-1],ys[-1],s = 200,marker="x", color = color,alpha=alpha)
    ax.scatter(xs,ys,s = 10, color = color,alpha=alpha)
    
    return ax

def plot_training(ax,state_history,
                  grid_shape,goal_state,
                  line_width=4,alpha=0.2,interp_spline_param = 2):
    ax.invert_xaxis()
    
    plot_grid(ax,grid_shape,goal_state)
    
    COLOR_START = np.array([0,0,1])
    COLOR_END = np.array([1,0,0])
    Ntrials = state_history.shape[0]
    
    ts = np.linspace(0,1,Ntrials)
    for trial,t in enumerate(ts):       
        color = COLOR_START + t*(COLOR_END - COLOR_START)
        
        plot_trajectory(ax,state_history[trial],grid_shape,color,lw=line_width,alpha=alpha)
        
        
    

if __name__ == "__main__":
    
    grid_shape = (7,7)
    goal_state = (0,6)
    
    xs = np.array(
        [[29, 36, 30, 24, 30, 24, 32, 24, 25 ,17,  9],
        [37, 44, 43, 36, 28, 22, 21, 21, 21 ,21 ,21]]
    )
    
    # xs = np.ones((20,))
    
    fig,ax = plt.subplots(1)
    
    plot_training(ax,xs,grid_shape,goal_state)
    
    # plot_grid(ax,grid_shape)
    # plot_trajectory(ax,xs,grid_shape)
    
    fig.show()
    input()