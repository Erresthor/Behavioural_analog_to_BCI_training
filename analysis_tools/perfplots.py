


import sys,os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly as pltly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.mixture import GaussianMixture
import joypy

from jax import vmap
import jax.numpy as jnp

# + local functions : 
from database_handling.database_extract import get_all_subject_data_from_internal_task_id
from utils import remove_by_indices
from analysis_tools.preprocess import get_preprocessed_data_from_df

NOISE_COLORMAP = [[c/256.0 for c in col] for col in [[10, 51, 204],[179, 0, 134],[230, 0, 0]]]

def perfplot(plot_this_dataframe,plot_type='violin',xcat_name="noise_category"):

    means = plot_this_dataframe.groupby(xcat_name)['final_performance'].mean()



    reduced_df = plot_this_dataframe[[xcat_name,"performance_category"]]
    # Count the occurrences of each combination
    counts = reduced_df.value_counts().reset_index()
    counts.columns = [xcat_name, "performance_category", "counts"]
    # Compute the total number of subjects per Trial Group
    total_per_group = counts.groupby(xcat_name)["counts"].transform("sum")
    # Normalize counts (convert to proportion)
    counts["Proportion"] = counts["counts"] / total_per_group

    if xcat_name == "noise_category" :
        label = "Feedback noise category"
        
        color_dict = {str(cat): np.array(NOISE_COLORMAP[k]) for k,cat in enumerate(["Low","Medium","High"])}
        
    elif xcat_name == "feedback_noise_std":
        label = "Feedback noise standard deviation"

        color_dict = {str(cat): np.array(NOISE_COLORMAP[k]) for k,cat in enumerate([0.025,0.1,0.175])}


    fig,axs = plt.subplots(1,2,figsize = (14,7),dpi=150)
    # fig.suptitle("Effect of feedback noise on final performance \n (study 1)")

    ax = axs[0]
    colors = ['peachpuff', 'orange', 'tomato']
    ax.set_title("Effect of feedback noise\non final performance")
    if plot_type == 'violin' : 
        sns.violinplot(ax=ax,x=xcat_name, y="final_performance",
                            data=plot_this_dataframe, palette=color_dict,
                            scale="area", inner="stick",
                            scale_hue=True, bw=.2)
    elif plot_type == 'boxplot' : 
        sns.boxplot(ax=ax,x=xcat_name, y="final_performance",
                    data=plot_this_dataframe, palette=color_dict)
        
        # Adjust zorder and label
        # ax.scatter(np.arange(len(means)), means, color='black', label='Mean', zorder=5, s=100, marker='D')
        # Adding a legend to label the means
        # add stripplot to boxplot with Seaborn
        sns.stripplot(ax = ax, y='final_performance', x=xcat_name, 
                        data=plot_this_dataframe, 
                        jitter=True, 
                        marker='o', 
                        alpha=0.2,
                        color='black')
    ax.scatter(np.arange(len(means)), means, color='black', label='Mean', zorder=5, s=100, marker='D')
    ax.set_xlabel(label)
    ax.set_ylabel("$Pf$ (Final performance)")
    ax.legend(loc='lower center')


    custom_order = ["Poor","Middling","Good"]
    ax_2 = axs[1]
    ax_2.set_title("Subject performance category ratio \n depending on feedback noise")
    sns.barplot(ax=ax_2,data=counts, x=xcat_name, y="Proportion", hue="performance_category",hue_order=custom_order)
    plt.legend(title="Performance category", fontsize=16, loc="upper right")

    total_noise_params = plot_this_dataframe[xcat_name].value_counts()
    for i, nc in enumerate(total_noise_params.index):
        total = total_noise_params[nc]
        ax_2.text(i, 0.66, f"N={total}", ha="center", va="bottom", fontsize=18)
    
    
    ax_2.set_xlabel(label)
    # ax_2.xaxis.labelpad = 20
    ax_2.set_ylabel("Proportion of subjects")
    

    for ax in axs :
        ax.grid()
        ax.set_ylim([0,1])
        
    fig.show()

def joyplot(plot_this_dataframe):
    eps = 0.2
    tfin = 5

    # Calculate the mean and std of the feedabck and (1- euclidian distance) arrays for each group
    all_distances = np.stack(plot_this_dataframe['norm_distance_to_goal'])[:,:-1,:]
    Nsubjs = plot_this_dataframe.shape[0]

    # Only look at the last timesteps avg : 
    all_distances = np.mean(all_distances[:,:,-tfin:],axis=-1)
    Nsteps = all_distances.shape[-1]

    all_subject_indexes = np.repeat(np.expand_dims(np.arange(Nsubjs),1),Nsteps,1)
    all_trials_indexes = np.repeat(np.expand_dims(np.arange(Nsteps),0),Nsubjs,0)
    all_subject_noise_cat = np.repeat(np.expand_dims(np.array(plot_this_dataframe['noise_category']),1),Nsteps,1)
    all_noises = np.repeat(np.expand_dims(np.array(plot_this_dataframe['feedback_noise_std']),1),Nsteps,1)

    # Dataframe : columns : step x subject_id x noise_level x final_feedback x final_distance
    new_df = pd.DataFrame()
    new_df["trial"] = list(all_trials_indexes.flatten())
    new_df["subject"] = list(all_subject_indexes.flatten())
    new_df["block_performance"] = list((1.0-all_distances).flatten())
    new_df["final_distance"] = list(all_distances.flatten())
    new_df["noise"] = list(all_noises.flatten())
    new_df['noise_cat'] = list(all_subject_noise_cat.flatten())


    BB = new_df.pivot_table(index=['subject','trial'], columns='noise_cat', values='block_performance').reset_index()



    fig, axs = joypy.joyplot(BB, column=["Low", "Medium", "High"],color=[tuple(col) for col in NOISE_COLORMAP], 
                            by="trial", ylim='own', 
                            figsize=(10,14) ,
                            alpha=0.7, legend=True,fill=False, linewidth=5)
    
    # Decoration
    plt.title('Subject Block performance KDE \n evolution across blocks \n depending on FB noise', fontsize=32, alpha=0.9)
    plt.rc("font", size=20)
    
    ax = axs[-1]
    ax.yaxis.set_label_coords(-0.1, .5)
    ax.set_xlabel('$Pb$ (Block performance)',  fontsize=16,alpha=1)
    ax.set_ylabel("Blocks")
    ax.yaxis.set_visible(True)
    ax.yaxis.set_ticks([])
    ax.axvline(0,color="black")
    ax.axvline(1,color="black")
    for ax in axs : 
        ax.axhline(0,color="grey")
    fig.show()

def perfplot_both(plot_this_dataframe,xcat_name="noise_category"):
        
    means = plot_this_dataframe.groupby(xcat_name)['final_performance'].mean()

    if xcat_name == "noise_category" :
        label = "Feedback noise category"
        
        color_dict = {str(cat): np.array(NOISE_COLORMAP[k]) for k,cat in enumerate(["Low","Medium","High"])}
        
    elif xcat_name == "feedback_noise_std":
        label = "Feedback noise standard deviation"

        color_dict = {str(cat): np.array(NOISE_COLORMAP[k]) for k,cat in enumerate([0.025,0.1,0.175])}
    
    fig,axs = plt.subplots(1,2,sharey=True,figsize = (10,7))
    fig.suptitle("Effect of feedback noise on final performance")
    ax = sns.violinplot(ax=axs[0],x=xcat_name, y="final_performance",
                        data=plot_this_dataframe, palette=color_dict,
                        scale="area", inner="stick",
                        scale_hue=True, bw=.2)
    ax.scatter(np.arange(len(means)), means, color='red', label='Mean', zorder=5, s=100, marker='D')

    ax = sns.boxplot(ax=axs[1],x=xcat_name, y="final_performance",
                        data=plot_this_dataframe, palette=color_dict)

    # Adjust zorder and label
    ax.scatter(np.arange(len(means)), means, color='red', label='Mean', zorder=5, s=100, marker='D')
    # Adding a legend to label the means
    ax.legend()

    # add stripplot to boxplot with Seaborn
    sns.stripplot(ax = axs[1], y='final_performance', x=xcat_name, 
                    data=plot_this_dataframe, 
                    jitter=True, 
                    marker='o', 
                    alpha=0.2,
                    color='black')

    for ax in axs :
        ax.grid()
        ax.set_ylim([0,1])
    fig.show()