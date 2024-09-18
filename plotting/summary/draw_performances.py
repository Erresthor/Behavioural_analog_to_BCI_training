import pymongo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from itertools import chain

def get_first_value_in_actual_expe(array_of_fbs,trial_start):
    for k in range(array_of_fbs.shape[0]):
        if (array_of_fbs[k,0] > trial_start) :
            return k

def draw_scores(ax,trial_scores,timestep_scores,trial_starts,trial_ends,gauge_animation_ends):
    offset = 0

    for i in range(len(trial_starts)):
        rect = patches.Rectangle((trial_starts[i],offset), (trial_ends[i]-trial_starts[i]),(trial_scores[i]/100.0),facecolor=(0.1, 0.2, 0.5, 0.3))
        ax.add_patch(rect)
        
        time_when_new_feedback_appears = (np.concatenate((np.array([trial_starts[i]]),np.array(gauge_animation_ends[i]))))
        # print(np.concatenate(np.array(gauge_animation_ends[i]),np.array([trial_ends[i]])))
        # print(time_when_new_feedback_appears)
        # print((timestep_scores[i,:]+offset))
        ax.plot(time_when_new_feedback_appears,(timestep_scores[i,:]+offset)[:time_when_new_feedback_appears.shape[0]], marker='x',color='black')


def draw_rt_fb(ax,fb_rtv,t0=0,tf=-1):
    offset =0
    ax.scatter(fb_rtv[t0:tf,0],fb_rtv[t0:tf,1]+offset,color="blue",marker="x",s=10,label="value")
    ax.scatter(fb_rtv[t0:tf,0],fb_rtv[t0:tf,2]+offset,color="red",marker="x",s=10,label="true value")


def draw_perf(trial_scores,timestep_scores,trial_starts,trial_ends,gauge_anim_ends,
              rt_fb_arr):
    # PLOT TIMELINE & FEEDBACK
    fig, ax = plt.subplots(figsize=(12,3))
    draw_scores(ax,trial_scores,timestep_scores,trial_starts,trial_ends,gauge_anim_ends)
    draw_rt_fb(ax,rt_fb_arr)
    return fig