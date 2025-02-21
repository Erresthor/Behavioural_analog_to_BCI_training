import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline
from .simulate_utils import ind2sub

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
                    lw = 4,alpha=0.2,sigma = 0.1):
    
    xs,ys = ind2sub((grid_shape),state_idxs_trial)
    
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
                  line_width=4,alpha=0.2,interp_spline_param = 2,sigma = 0.1):
    ax.invert_xaxis()
    
    plot_grid(ax,grid_shape,goal_state)
    
    COLOR_START = np.array([0,0,1])
    COLOR_END = np.array([1,0,0])
    Ntrials = state_history.shape[0]
    
    ts = np.linspace(0,1,Ntrials)
    for trial,t in enumerate(ts):       
        color = COLOR_START + t*(COLOR_END - COLOR_START)
        
        plot_trajectory(ax,state_history[trial],grid_shape,color,lw=line_width,alpha=alpha,sigma=sigma,interp_spline_param=interp_spline_param)
        
def plot_trial(ax,state_history,
                  grid_shape,goal_state,
                  line_width=4,alpha=0.2,interp_spline_param = 2,
                  color=np.array([1.0,0.5,0.5])):
    ax.invert_xaxis()
    
    plot_grid(ax,grid_shape,goal_state)

    
    plot_trajectory(ax,state_history,grid_shape,color,lw=line_width,alpha=alpha)  
    
    

def plot_actions(ax,_agent_actions,label,
                 plotstyle = 'lines',
                    noise_scale=  5e-2):
    Ntimesteps,_ =_agent_actions["angle"].shape
    
    def action_index_to_value(modality,index,Nsamples=1):
        step_factor = noise_scale
        if modality == "angle":
            if index == 0 :
                return None,[None for n in range(Nsamples)]
                
            sigma = step_factor*(1/8)*2*np.pi 
            mu = (index-1)*(1/8)*2*np.pi
            
            return mu, np.random.normal(mu, sigma, size=Nsamples)
            
        elif modality == "position" :
            
            sigma = step_factor*(1/3) 
            cov = [[sigma,0],[0,sigma]]
            
            (y,x) = ind2sub((3,3),index)
            xgrid = 0.5*(1/3) + x*(1/3)
            ygrid = 0.5*(1/3) + y*(1/3)
            mu = np.array([xgrid,ygrid])
            return mu, np.clip(np.random.multivariate_normal(mu, cov, size=Nsamples),a_min=0,a_max=1)
            
        elif modality == "distance" :
            min_dist_norm = 7.5/(np.sqrt(2)*750)
            distance_bins = np.array([0.0,min_dist_norm,0.2,0.5,np.sqrt(2) + 1e-10])
            sigmas = [step_factor*(distance_bins[i+1]-distance_bins[i]) for i in range(len(distance_bins)-1)]
                    # Half the distance between both boundaries
                    
            mu = 0.5*np.mean(np.array([distance_bins[index],distance_bins[index+1]]))
            sigma = sigmas[index]
            
            return  mu, np.clip(np.random.normal(mu, sigma, size=Nsamples),a_min=0,a_max=np.sqrt(2))


    def rotation_matrix_2d(theta):
        """
        Returns a 2D rotation matrix for a given angle in radians.

        Parameters:
        theta (float): Rotation angle in radians.

        Returns:
        numpy.ndarray: 2x2 rotation matrix.
        """
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]]).squeeze()
        


    def action_to_points(action_dict):
        angle_mu,angle_sample = action_index_to_value("angle",action_dict["angle"],Nsamples=1)
        if angle_mu is None:
            angle_sample = np.random.random()*np.pi*2
        
        position_mu,position_sample = action_index_to_value("position",action_dict["position"],Nsamples=1)    
        distance_mu,distance_sample = action_index_to_value("distance",action_dict["distance"],Nsamples=1)
        
        x1 = position_sample + (distance_sample/2.0)*np.dot(rotation_matrix_2d(-angle_sample+np.pi),np.array([1,0]))
        x2 = position_sample + (distance_sample/2.0)*np.dot(rotation_matrix_2d(-angle_sample),np.array([1,0]))
        return np.stack([x1,x2]).squeeze(),position_mu

    # Draw the grid :


    def cosine_similarity(a, b):
        cos_simimarity =  (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)+0.00001))
        return (cos_simimarity + 1)/2

    def dict_vect_to_ind(_dict):
        return {key:np.argmax(val) for (key,val) in _dict.items()}

    ax.set_title(label)
    eps = 0.2
    ax.set_ylim([1+eps,0-eps])
    ax.set_xlim([0-eps,1+eps])
    ax.hlines(0,xmin=0,xmax=1,color="black",lw=2)
    ax.vlines(0,ymin=0,ymax=1,color="black",lw=2)
    ax.hlines(1,xmin=0,xmax=1,color="black",lw=2)
    ax.vlines(1,ymin=0,ymax=1,color="black",lw=2)
    ax.set_axis_off()

    for action in range(Ntimesteps) :
        plotted_dict = {key:val[action,:] for (key,val) in _agent_actions.items()}
        
        Xs, pos_mu= action_to_points(dict_vect_to_ind(plotted_dict))
        
        if plotstyle=='dots':
            if (np.linalg.norm(Xs[1] - Xs[0])<0.02): 
                ax.scatter(Xs[1,0],Xs[1,1],s=100,alpha=0.4,color="black")   
            else :
                ax.plot(Xs[:,0],Xs[:,1],lw=1,linestyle="--",color='grey')
                ax.scatter(Xs[0,0],Xs[0,1],color="blue")
                ax.scatter(Xs[1,0],Xs[1,1],color="red")
            
        elif plotstyle =='lines':
            t = cosine_similarity(Xs[1] - Xs[0],np.array([100,-100]))
            col = np.array([1,0,0])*t + (1-t)*np.array([0,0,1])
            
            
            
            
            if (np.linalg.norm(Xs[1] - Xs[0])<0.02): 
                ax.scatter(Xs[1,0],Xs[1,1],s=100,alpha=0.4,color="black")   
            else :
                ax.plot(Xs[:,0], Xs[:,1],'-', lw=5,alpha=0.7,color=col)
        # ax.scatter(pos_mu[0],pos_mu[1],color="green")



def plot_learnt_transition_matrix(states,trial,timestep):
    fig,axs = plt.subplots(3,9)
    for k,(modality) in enumerate(["position","angle","distance"]):
        bmod = states["B"][modality][trial,timestep]
        for u in range(9):
            if u < bmod.shape[-1]:
                axs[k,u].imshow(bmod[...,u],vmin=0,vmax=1)
            else :
                axs[k,u].axis('off')
    fig.show()



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