


# from actynf.jaxtynf.layer_options import get_planning_options,get_action_selection_options,get_learning_options
# from test_models_jax import behavioural_process,naive_model,basic_latent_model,grid_latent_model

# # DEFINING THE ENVIRONMENT : 
# N_fb_ticks = 5
# trueA,trueB,trueC,trueD,trueE,trueU = behavioural_process(GRID_SIZE,START_COORD,END_COORD,N_fb_ticks)
# # print(A[0].shape)
# # print(B[0].shape)

# def get_task_models(model_parameters,modeltype="naive"):    
    
#     if modeltype=="naive":
#         model_builder = naive_model
#     elif modeltype=="1d":
#         model_builder = basic_latent_model
#     elif modeltype=="2d":
#         model_builder = grid_latent_model
#     else:
#         raise NotImplementedError("Model type not implemented")
    
#     models = {}
#     for action_model in ["angle","position","distance"]:
#         a,b,c,d,e,u = model_builder(model_parameters,action_model)
#         models[action_model] = [a,b,c,d,e,u]
    
    
#     # Options are shared across all model types & action modalities ?
#     generalize_fadeout = lambda x : jnp.exp(-model_parameters["generalization_inv_temp"]*x)
    
#     planning_options = get_planning_options(Th,planning_method = "sophisticated",
#                         state_horizon = model_parameters["N_state_branches"],
#                         action_horizon=model_parameters["N_action_branches"],
#                         explore_remaining_paths=model_parameters["explore_remaining"],
#                         a_novel=False,b_novel=False,
#                         old_efe_computation=True)
#     action_selection_options = get_action_selection_options("stochastic",model_parameters["alpha"])
#     learning_options = get_learning_options(learn_a=False,learn_b=True,learn_d=True,lr_b = model_parameters["lr_b"],lr_d = model_parameters["lr_d"],run_smoother=True,assume_linear_state_space=True,gen_fadeout_func=generalize_fadeout)
         
#     return models,planning_options,action_selection_options,learning_options


# parameters = {
#     # General environment : 
#     "N_feedback_ticks":N_fb_ticks,  
    
#     # Model structure
#     "Ns_latent":5,  # For 1D
#     "grid_size":[7,7],  # For 2D
    
#     # Model parameters
#     "b_str_init":0.1,
#     "d_str_init":0.1,
#     "goal_pos_str":1.0, # For the grid model only
#     "reward_seeking" : 10.0,
    
#     # Planning 
#     "N_state_branches" : 2,
#     "N_action_branches" : 9,
#     "explore_remaining" : False,
    
#     # Action selection 
#     "alpha" : 4.0,
    
#     # Learning
#     "generalization_inv_temp":0.5,    
#     "lr_b" : 1.0,
#     "lr_d" : 1.0
# }

# models,opt_plan,opt_select,opt_learn = get_task_models(parameters,"2d")

# [a0,b0,c,d0,e,u] = models["angle"]
# # print(a)



# print(opt_plan)
# print(to_log_space(c,e))()



# # Running the model: (First 10 trials)
# Ntrials = 10
# rngkey = jax.random.PRNGKey(10)
# print(opt_plan)
# [all_obs_arr,all_true_s_arr,all_u_arr,
#  all_qs_arr,all_qs_post,
#  all_qpi_arr,efes_arr,
#  a_hist,b_hist,d_hist] = synthetic_training(rngkey,
#             Ntrials,T,
#             trueA,trueB,trueD,trueU,
#             a0,b0,c,d0,e,u,
#             planning_options = opt_plan,
#             action_selection_options = opt_select,
#             learning_options = opt_learn)