import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial

from actynf.jaxtynf.jax_toolbox import tensorify

from actynf.base.function_toolbox import normalize
from .simulate_utils import sub2ind,ind2sub,distance,discretized_distribution_from_value,discretize_normal_pdf

from jax.lax import stop_gradient


def to_jax(x):
    return stop_gradient(tensorify(x))

def behavioural_process(grid_size,start_idx,end_idx,n_feedback_ticks,feedback_std):
    """One dimension latent space environment

    Args:
        grid_size (_type_): _description_
        start_idx (_type_): _description_
        end_idx (_type_): _description_
        n_feedback_ticks (_type_): _description_
        feedback_std (_type_): _description_

    Returns:
        _type_: _description_
    """
    # TODO : update it to fit the conventions of the data extraction 
    
    flattened_grid = jnp.zeros(grid_size).flatten()
    Ns = flattened_grid.shape[0]

    # Starting position
    d0 = np.zeros((Ns,))
    if (type(start_idx)==list):
        start_pos = [sub2ind(grid_size,ix) for ix in start_idx]
        for pos in start_pos : 
            d0[pos] = 1.0
    else:
        start_pos = sub2ind(grid_size,start_idx)
        d0[start_pos] = 1.0
    d = tensorify([normalize(d0)])    
    
    
    
    all_scalar_fb_values = np.zeros((Ns,))
    for idx,state in enumerate(flattened_grid):
        cellx,celly = ind2sub(grid_size,idx)
        distance_to_goal = distance([cellx,celly],end_idx,True,grid_size[0])
        all_scalar_fb_values[idx] = 1.0 - distance_to_goal
    
    discretize_distance_normal_function = partial(discretize_normal_pdf,std=feedback_std,num_bins = n_feedback_ticks,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
    a0,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
    a = [a0]

    # Transition matrices
    # A lot of possible actions, we discretize them as follow : 
    # 9 possible angles x 9 possible mean positions x 3 possible distances = 243 possible actions
    # To sample efficiently, we assume that subjects  entertain different hypotheses regarding what actions
    # affect the feedback, that they sample simultaneously
    
    # In practice, only the angle mapping matters :
    # Angle mappings : 
    angle_maps = [[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0], [-1,1]]
    #angle degrees :NA,   0 ,   45,   90,   135,  180,    225,  270,   315
    
    # Warning ! VERTICAL axis (x coords) is inverted for numpy arrays !
    B_angle = np.zeros((Ns,Ns,len(angle_maps)))
    for from_state in range(Ns):
        from_x,from_y = ind2sub(grid_size,from_state)
        for action in range(B_angle.shape[-1]):
            to_x = from_x - angle_maps[action][0] # Inversion : going up is lowering your line value
            to_y = from_y + angle_maps[action][1]  
            
            if ((to_x<0) or (to_x>=grid_size[0])) : 
                to_x = from_x
            
            if ((to_y<0) or (to_y>=grid_size[1])) :
                to_y = from_y

            to_state = sub2ind(grid_size,(to_x,to_y))
            
            B_angle[to_state,from_state,action]= 1.0

    # All the other action modalities are neutral w.r.t the hidden states
    # i.e. their state to state mapping is an identity matrix
    B_mean_pos = np.zeros((Ns,Ns,9))
    for action in range(B_mean_pos.shape[-1]):
        B_mean_pos[:,:,action] = np.eye(Ns)
    
    B_distances = np.zeros((Ns,Ns,3))
    for action in range(B_distances.shape[-1]):
        B_distances[:,:,action] = np.eye(Ns)


    # To simplify, let's assume that only the angle input is connected to the process :
    b = [B_angle]
    u = np.array(range(b[0].shape[-1])) # allowable actions

    # We receive action outputs from 3 differents sources
    c = [np.linspace(0,n_feedback_ticks-1,n_feedback_ticks)]

    e = np.ones(u.shape)
    # maze_process = mdp_layer("beh_process","process",a,b,c,d,e,u,T,Th,in_seed=seed)
    
    
    # Make into jax tensors
    return_these = [to_jax(x) for x in [a,b,c,d,e,u]]
    return tuple(return_these),all_scalar_fb_values


def naive_model(parameters,action_model="angle"):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    # MAIN ASSUMPTION : HIDDEN STATES ARE DIRECTLY OBSERVABLE USING THE FEEDBACK
    n_feedback_ticks = parameters["N_feedback_ticks"]
    Ns = n_feedback_ticks
    
    # Initial state belief :
    initial_state_concentration = parameters["initial_state_concentration"]
    d0 = initial_state_concentration*jnp.ones((Ns,))
        # Uninformed prior on hidden state position before the task starts
    d = [d0]

    # Emissions :
    # The feedback is a one-dimensionnal information related to the cognitive dimensions
    # Assume that the feedback seen is directly related to the hidden state (no hidden state)
    a = [jnp.eye(n_feedback_ticks)]

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    if (action_model=="angle"):
        n_possible_actions = parameters["N_actions_angle"]
    elif(action_model=="position"):
        n_possible_actions = parameters["N_actions_position"]
    elif(action_model=="distance"):
        n_possible_actions = parameters["N_actions_distance"]
    
    # An initially naive model !
    transition_stickiness = parameters["transition_stickiness"]
    transition_concentration = parameters["transition_concentration"]
    b0 = transition_concentration*jnp.ones((Ns,Ns,n_possible_actions)) + transition_stickiness*jnp.expand_dims(jnp.eye(Ns),-1)
    b = [b0]

    # Assume a linear preference matrix c = ln p(o)
    rs = parameters["reward_seeking"]
    c = [jnp.linspace(0,rs,n_feedback_ticks)]

    u = to_jax(np.expand_dims(np.array(range(b[0].shape[-1])),-1))
    
    e = jnp.ones(u.shape)

    return a,b,c,d,e,u

def basic_latent_model(parameters):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    
    # MAIN ASSUMPTION : HIDDEN STATES ARE *NOT* DIRECTLY OBSERVABLE USING THE FEEDBACK
    # BUT THE FEEDBACK GIVES AN ESTIMATE OF THE CURRENT LATENT STATE VALUE
    n_feedback_ticks = parameters["N_feedback_ticks"]
    Ns = parameters["Ns_latent"]
    
    
    
    # Initial state belief :
    initial_state_concentration = parameters["initial_state_concentration"]
    d0 = initial_state_concentration*jnp.ones((Ns,))
        # Uninformed prior on hidden state position before the task starts
    d = [d0]



    # Emissions :
    feedback_std = parameters["feedback_expected_std"]
    emission_stickiness = parameters["emission_stickiness"]
    emission_concentration = parameters["emission_concentration"]
    
    # The feedback is a one-dimensionnal information related to the latent state
    # This is first defined using numpy, but we will derive a jax implementation if it becomes needed later
    # (i.e. we parametrize part of this mapping)
    all_scalar_fb_values = np.linspace(0,1,Ns)   # Assume that the bigger the index of the state, the better the feedback
    discretize_distance_normal_function = partial(discretize_normal_pdf,std=feedback_std,num_bins = n_feedback_ticks,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
    stickiness,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
       
    # Perceptive prior is built as follows :
    # - a "base term" that represents subject uncertainty regarding the meaning of the feedback
    # - a "stickiness term" that represents subject prior belief that the feedback gives information about 
    #        its latent state
    base = jnp.ones((n_feedback_ticks,Ns))        
    
    a0 = emission_concentration*base + emission_stickiness*stickiness
    a = [a0]
    
    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    n_possible_actions = parameters["N_actions"]
    
    # An initially naive model !
    transition_stickiness = parameters["transition_stickiness"]
    transition_concentration = parameters["transition_concentration"]
    b0 = transition_concentration*jnp.ones((Ns,Ns,n_possible_actions)) + transition_stickiness*jnp.expand_dims(jnp.eye(Ns),-1)
    b = [b0]
    
    # Assume a linear preference matrix c = ln p(o)
    
    rs = parameters["reward_seeking"]
    c = [jnp.linspace(0,rs,n_feedback_ticks)]

    u = to_jax(np.expand_dims(np.array(range(b[0].shape[-1])),-1))
    e = jnp.ones((b[0].shape[-1],))

    return a,b,c,d,e,u



def simple_1D_model(parameters):
    '''
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    
    # MAIN ASSUMPTION : HIDDEN STATES ARE *NOT* DIRECTLY OBSERVABLE USING THE FEEDBACK
    # BUT THE FEEDBACK GIVES AN ESTIMATE OF THE CURRENT LATENT STATE VALUE
    n_feedback_ticks = parameters["N_feedback_ticks"]
    Ns = parameters["Ns_latent"]
    
    
    
    # Initial state belief :
    initial_state_concentration = 1.0
    d0 = initial_state_concentration*jnp.ones((Ns,))
        # Uninformed prior on hidden state position before the task starts
    d = [d0]



    # Emissions :
    feedback_std = parameters["feedback_expected_std"]
    emission_stickiness = 1.0
    emission_concentration = 0.0
    
    # The feedback is a one-dimensionnal information related to the latent state
    # This is first defined using numpy, but we will derive a jax implementation if it becomes needed later
    # (i.e. we parametrize part of this mapping)
    all_scalar_fb_values = np.linspace(0,1,Ns)   # Assume that the bigger the index of the state, the better the feedback
    
    discretize_distance_normal_function = partial(discretize_normal_pdf,std=feedback_std,num_bins = n_feedback_ticks,lower_bound= -1e-5 ,upper_bound = 1.0 + 1e-5)
    stickiness,edges = vmap(discretize_distance_normal_function,out_axes=-1)(all_scalar_fb_values)
       
    # Perceptive prior is built as follows :
    # - a "base term" that represents subject uncertainty regarding the meaning of the feedback
    # - a "stickiness term" that represents subject prior belief that the feedback gives information about 
    #        its latent state
    base = jnp.ones((n_feedback_ticks,Ns))        
    
    a0 = emission_concentration*base + emission_stickiness*stickiness
    a = [a0]
    
    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    n_possible_actions = parameters["N_actions"]
    
    # An initially naive model !
    transition_stickiness = parameters["initial_transition_stickiness"]
    transition_concentration = parameters["initial_transition_confidence"]
    b0 = transition_concentration*(jnp.ones((Ns,Ns,n_possible_actions)) + transition_stickiness*jnp.expand_dims(jnp.eye(Ns),-1))
    b = [b0]
    
    # Assume a linear preference matrix c = ln p(o)
    
    rs = parameters["reward_seeking"]
    c = [jnp.linspace(0,rs,n_feedback_ticks)]

    u = to_jax(np.expand_dims(np.array(range(b[0].shape[-1])),-1))
    e = jnp.ones((b[0].shape[-1],))

    return a,b,c,d,e,u








def grid_latent_model(parameters,action_model="angle"):
    '''
    The jist of this approach ! A true grid like model that explicitely models the position 
    as well as the target.
    
    MAIN ASSUMPTION : THERE ARE 4 HIDDEN STATES !
    1. x current position
    2. y current position
    3. x goal 
    4. y goal 
    
    3 and 4 are static for the duration of the trial
    1 and 2 
    
    A function defining a generic neurofeedback model depending on a few criteria:
    cognitive_layout : a list of the cognitive dimensions we wish to model
    feedback_Nticks : how we wish to discretize the feedback
    '''
    
    n_feedback_ticks = parameters["N_feedback_ticks"]
    
    grid_size = parameters["grid_size"]
    Ns = [grid_size[0],grid_size[1],grid_size[0],grid_size[1]]

    
    # Initial state belief :
    # Uninformed prior on hidden state position before the task starts
    # We start oblivious to the starting state!
    initial_state_concentration = parameters["initial_state_concentration"]
    d = [initial_state_concentration*np.ones((s,)) for s in Ns ]
    
    
    # # ASSUME INITIAL ORIENTATION (TO AVOID SYMMETRY-INDUCED AMBIGUITY)
    d[0][:int(d[0].shape[0]/2)] += 10.0
    d[1][int(d[1].shape[0]/2):] += 10.0
    # d[2][-1] += 20000.0
    # d[3][-1] += 20000.0
    d = to_jax(d)
    
    # ASSUME THAT WE REMAIN UNSURE OF WHERE THE GOAL IS 
    # d[2] = d[2]*str_goal_pos
    # d[3] = d[3]*str_goal_pos
    
    
    # Emissions :
    # Here, the subject assumes the feedback
    # is a one-dimensionnal information related to
    # the linear (L2) distance between my position and the goal !
    base = np.ones((n_feedback_ticks,) + tuple(Ns))
    informative_prior_raw_vals =  np.zeros(tuple(Ns))
    informative_prior = np.zeros((n_feedback_ticks,) + tuple(Ns))
    for x in range(Ns[0]):
        for y in range(Ns[1]):
            for xgoal in range(Ns[2]):
                for ygoal in range(Ns[3]):
                    expected_linear_feedback = 1.0-distance((x,y),(xgoal,ygoal),True,grid_size[0])
                    
                    feedback_dist = discretized_distribution_from_value(expected_linear_feedback,n_feedback_ticks)
                    
                    informative_prior_raw_vals[x,y,xgoal,ygoal] = expected_linear_feedback
                    informative_prior[:,x,y,xgoal,ygoal] = feedback_dist
                    
    
    emission_stickiness = parameters["emission_stickiness"]
    emission_concentration = parameters["emission_concentration"]
    a0 = emission_concentration*base + emission_stickiness*to_jax(informative_prior)
    a = [a0]
    
    # # Show what level of feedback we get depending if the goal state is at (0,6)
    # import matplotlib.pyplot as plt
    # img = np.zeros((Ns[0],Ns[1]))
    # img = feedback_raw_vals[:,:,0,4]
    # plt.imshow(img)
    # plt.show()

    # Action modalities : there are various ways of modeling those actions. 
    # Here, we consider that the subject may entertain 3 different models : 
    # 1. A model where the angle drives the feedback  (9 actions)
    # 2. A model where the position of the point drives the feedback (9 actions)
    # 3. A model where the distance between points drives the feedback (3 actions)
    if (action_model=="angle"):
        n_possible_actions = parameters["N_actions_angle"]
    elif(action_model=="position"):
        n_possible_actions = parameters["N_actions_position"]
    elif(action_model=="distance"):
        n_possible_actions = parameters["N_actions_distance"]
    
    # An initially naive model 
    transition_stickiness = parameters["transition_stickiness"]
    transition_concentration = parameters["transition_concentration"]
    b0 = transition_concentration*jnp.ones((Ns[0],Ns[0],n_possible_actions)) # + transition_stickiness*jnp.expand_dims(jnp.eye(Ns[0]),-1)
    b1 = transition_concentration*jnp.ones((Ns[1],Ns[1],n_possible_actions)) # + transition_stickiness*jnp.expand_dims(jnp.eye(Ns[1]),-1)
    
    # Biais time ! 
    if (action_model=="angle"):
        # There is a biais towards horizontal transitions for actions 0-4 
        # and towards vertical actions for actions 2-6
        
        # The "stay the same" prior is stronger on the dimensions where we don't expect much change :
        all_actions = jnp.arange(n_possible_actions)
        
        radial_subdivision = 2*jnp.pi/(n_possible_actions)
        
        prior_horizontal = jnp.abs(jnp.cos(all_actions*radial_subdivision))
            # Big values means we expect this to affect the horizontal axis
            # Low values means we expect this NOT to affect the horizontal axis, 
            # meaning that our prior that it will "stay the same" is bigger
        b0 = b0 + transition_stickiness*jnp.einsum("ij,w->ijw",jnp.eye(Ns[0]),1.0-prior_horizontal)
        
        prior_vertical = jnp.abs(jnp.sin(all_actions*radial_subdivision))
            # Big values means we expect this to affect the vertical axis
            # Low values means we expect this NOT to affect the vertical axis, 
            # meaning that our prior that it will "stay the same" is bigger
        b1 = b1 + transition_stickiness*jnp.einsum("ij,w->ijw",jnp.eye(Ns[1]),1.0-prior_vertical)
        
    elif(action_model=="position"):
        # There is a biais towards horizontal transitions for actions 0-4 
        # and towards vertical actions for actions 2-6
        l = 0
        
        
    elif(action_model=="distance"):
        n_possible_actions = parameters["N_actions_distance"]
        
        
    
    
    
    # assume actions have no effect on the goal position !
    b2 = 1.0*jnp.ones((Ns[2],Ns[2],n_possible_actions)) + 20000*jnp.expand_dims(jnp.eye(Ns[2]),-1)
    b3 = 1.0*jnp.ones((Ns[3],Ns[3],n_possible_actions)) + 20000*jnp.expand_dims(jnp.eye(Ns[3]),-1)
    b = [b0,b1,b2,b3]


    # Assume a linear preference matrix c = ln p(o)
    rs = parameters["reward_seeking"]
    c = [jnp.linspace(0,rs,n_feedback_ticks)]

    u = np.zeros((n_possible_actions,len(Ns)))
    for act in range(n_possible_actions):
        u[act,:] = np.array([act,act,0,0])
    u = to_jax(u.astype(int))
    
    e = jnp.ones(u.shape[0])

    return a,b,c,d,e,u





if __name__=="__main__":
    # We get a model weights by defining a "parameters" object :
    aif_1d_constants = {
        # General environment : 
        "N_feedback_ticks":10,
        # Latent state space structure
        "Ns_latent":5,      # For 1D
        # Action discretization:
        "N_actions_distance" :3,
        "N_actions_position" :9,
        "N_actions_angle" :9,
        
        "Th" : 3
    }

    aif_1d_params = {    
        # ----------------------------------------------------------------------------------------------------
        # Model parameters : these should interact with the model components in a differentiable manner
        "transition_concentration": 1.0,
        "transition_stickiness": 1.0,
        "transition_learning_rate" : 1.0,
        "state_interpolation_temperature" : 1.0,
        
        "initial_state_concentration": 1.0,
        
        "feedback_expected_std" : 0.0003,
        "emission_concentration" : 1.0,
        "emission_stickiness" : 100.0,
        
        "reward_seeking" : 10.0,
        
        "action_selection_temperature" : 10.0,
    }
    
    for x in np.linspace(0.0005,0.1):
        aif_1d_params["feedback_expected_std"] = x
        a0,b0,c0,d0,e0,u = basic_latent_model({**aif_1d_constants, **aif_1d_params})
        print("------------------")
        print(x)
        print(a0)
