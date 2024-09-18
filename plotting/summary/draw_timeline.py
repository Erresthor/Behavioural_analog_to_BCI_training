import pymongo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from itertools import chain

ACTION_PLOT_POSITION = 2

# utils : to avoid issues when the event did not finish (incomplete database)
def get_start_and_end(listA,listB,k_elt,max_end=None):
    # Assuming listA[k] <= listB[k]
    start = listA[k_elt]
    try:
        end = listB[k_elt]
    except:
        if max_end == None:
            end = start 
        else : 
            end = max_end
            # raise NotImplementedError("Not implemented yet")
    return start,end

def draw_trials(ax,trial_starts,trial_ends,t_task_end):
    for k in range(len(trial_starts)):
        starting_time,ending_time = get_start_and_end(trial_starts,trial_ends,k,max_end=t_task_end)
        
        rect = patches.Rectangle((starting_time,0), (ending_time-starting_time),3,facecolor='grey',edgecolor='black',linewidth=2)
        ax.add_patch(rect)
        ax.annotate("TRIAL "+str(k),((starting_time+ending_time)/2,1),color='black', weight='bold', fontsize=10,ha='center', va='center')
        
    ax.scatter(trial_ends, np.full_like(trial_ends, 1), marker='o', s=100, color='red', edgecolors='black', zorder=3, label='trial_ends')

def drawaction_starts(ax,flatten_list_act,flatten_list_act_end,t_task_end):
    
    ax.scatter(flatten_list_act, np.full_like(flatten_list_act, ACTION_PLOT_POSITION), marker='^', s=150, color='green', edgecolors='black', zorder=3, label='actions')
    ax.scatter(flatten_list_act_end, np.full_like(flatten_list_act_end, ACTION_PLOT_POSITION), marker='^', s=50, color='green', edgecolors='black', zorder=3, label='actions')
    for k in range(len(flatten_list_act)):
        start_action,end_action = get_start_and_end(flatten_list_act,flatten_list_act_end,k,max_end=t_task_end)
        ax.plot([start_action,end_action],[ACTION_PLOT_POSITION,ACTION_PLOT_POSITION], "g--")

def drawgaugeanimend(ax,flatten_gauge_list):
    ax.scatter(flatten_gauge_list, np.full_like(flatten_gauge_list,4 ), marker='.', s=100, color='blue', edgecolors='black', zorder=3, label='actions')

def draw_events(ax,events,t_task_end):
    
    # INSTRUCTIONS:
    def y_from_instr(instr_stamp):
        if(instr_stamp == "A"):
            return 2
        elif (instr_stamp =="B"):
            return 3
        elif(instr_stamp =="TRY"):
            return 4
        elif (instr_stamp =="instructions"):
            return 
        else :
            return 1
    
    instr_trackers = []
    instruction_plots = []
    instr_lims = [0,0]
    
    # print(events["instructions"])
    # exit()
    # event_instr is always of the form : 
    # requested_instruction_X where X is a code
    for eventInstr in events["instructions"] :
        last_char = eventInstr["val"].split("_")[-1]
        instr_type = eventInstr["val"].split("_")[-2]
        success = ("success" in eventInstr["val"]) 
        ypos = y_from_instr(last_char)
        xpos = eventInstr["t"]
        
        if (last_char=="instructions"): # Instructions start or finish
                                        # Should only happen twice
            if(instr_type == "start"):
                instr_lims[0] = xpos
            else : 
                instr_lims[1] = xpos
         
        else :
            # if "prompt" in eventInstr["val"]:
                # # This is a prompt fail
                # print(eventInstr["val"])
                
            instr_trackers.append([xpos,instr_type+"_"+last_char,last_char,ypos,success])
            instruction_plots.append([xpos,ypos])
            ax.annotate(last_char,(xpos-200*len(last_char),ypos+0.5))
    
    # Show the instructions zone :
    instr_zone_y = -1
    if instr_lims[1]==0:
        instr_lims[1] = t_task_end
    rect = patches.Rectangle((instr_lims[0],instr_zone_y), (instr_lims[1]-instr_lims[0]),1,facecolor='lightgreen',edgecolor='green',linewidth=2)
    ax.add_patch(rect)
    
    instruction_plots = np.array(instruction_plots)
    ax.scatter(instruction_plots[:,0],instruction_plots[:,1], marker='o', s=100, color='blue', edgecolors='black', zorder=3, label='instructions')
    


    # SCREEN EVENTS:
    entered_events = []
    exit_events = []
    # screen_tracker = []
    for screen_event in events["fullscreen"]:
        entered = ("entered" in screen_event["val"])
        if entered :
            entered_events.append(screen_event["t"])
        else :
            exit_events.append(screen_event["t"])
    ax.scatter(entered_events,np.full_like(entered_events, -5), marker='s', s=250, color='green', edgecolors='black', zorder=3, label='entered_window')
    ax.scatter(exit_events,np.full_like(exit_events, -5), marker='s', s=250, color='red', edgecolors='black', zorder=3, label='exited_window')
    
    #CHARTS EVENTS
    charts_shown_tracker = []
    charts_hidden_tracker = []
    for chart_evt in events["charts"]:
        t = chart_evt["t"]
        chart_str = chart_evt["val"]
        if("shown" in chart_str):
            charts_shown_tracker.append(t)
        else : 
            charts_hidden_tracker.append(t) 
        
    for k in range(len(charts_shown_tracker)):
        rect = patches.Rectangle((charts_shown_tracker[k],1.5), (charts_hidden_tracker[k]-charts_shown_tracker[k]),0,facecolor='lightgrey',edgecolor='black',linewidth=2)
        ax.add_patch(rect)
        ax.annotate("CHART "+str(k),((charts_shown_tracker[k]+charts_hidden_tracker[k])/2,2.5),color='black', weight='bold', fontsize=10,ha='center', va='center')
    
    
    # WAIT EVENTS 
    wait_starts = []
    wait_ends  =[]
    wait_val = []
    started_waiting = False
    
    for waitEvent in events["wait"]:
        code = waitEvent["val"].split("_")[0]
        
        if started_waiting :
            assert code=="end", "The subject started waiting but never ended"
            
            wait_ends.append(waitEvent["t"])
            wait_val.append(waitEvent["val"].split("_")[1]) # Why waiting
            
            started_waiting = False
        else : 
            wait_starts.append(waitEvent["t"])
            started_waiting = True
    
    if started_waiting: # If we didnt get the final data, assume it happened till the end of the task
        print("The subject started waiting but never ended")
        wait_ends.append(t_task_end)
        wait_val.append(waitEvent["val"].split("_")[1]) # Why waiting (last value)
    
    for k in range(len(wait_ends)):
        rect = patches.Rectangle((wait_starts[k],0), (wait_ends[k]-wait_starts[k]),3,facecolor='lightblue')
        ax.add_patch(rect)
        ax.annotate(wait_val[k],((wait_starts[k]+wait_ends[k])/2,1),color='black', weight='bold', fontsize=5,ha='center', va='center')
    
    # ACTION SKIP EVENTS:
    # print(events["timesteps"])
    skips = []
    for skipEvent in events["timesteps"] :
        skips.append(skipEvent["t"])
    ax.scatter(skips,np.full_like(skips, ACTION_PLOT_POSITION), marker='^', s=150, color='red', edgecolors='black', zorder=3, label='actions MISSED')
    # For now, no timestep events

    #GAUGE ANIMATING EVENTS :  
    started_animating = False
    gau_starts = []
    gau_ends  =[]

    for gaugeEvent in events["gauge"]:
        is_an_animation_start = (gaugeEvent["val"].split("_")[1]=="start")
        if started_animating:
            # There should be either an end or a forceful end here : 
            if not(is_an_animation_start) : 
                # "Never stopped animating after " + str(gau_starts[-1])
                gau_ends.append(gaugeEvent["t"])
                started_animating = False
        else : 
            if (is_an_animation_start):
                gau_starts.append(gaugeEvent["t"])
                started_animating = True
    if started_animating: # If we didnt get the final data, assume it happened till the end of the task
        gau_ends.append(t_task_end)
        
    for k in range(len(gau_ends)):
        rect = patches.Rectangle((gau_starts[k],3), (gau_ends[k]-gau_starts[k]),1,facecolor='purple')
        ax.add_patch(rect)
    
    # FLASHES EVENTS : 
    # Assume the list is ordered
    fes = [0,0]
    for flashEvent in events["flashes"]:
        start =  (flashEvent["val"].split("_")[2]=="start")
        color = flashEvent["val"].split("_")[1]
        if (color=="white"):
            color = "gray"
        
        if (start):
            fes[0] = flashEvent["t"]
        else : 
            fes[1] = flashEvent["t"]
            rect = patches.Rectangle((fes[0],0), (fes[1]-fes[0]),1.5,facecolor=color)
            ax.add_patch(rect)


def draw_timeline(trial_starts,trial_ends,acts_starts,acts_ends,
                  events,end_clock):
    
    # PLOT TIMELINE & FEEDBACK
    # print(trial_ends,trial_starts)
    fig, ax = plt.subplots(figsize=(12,3))

    draw_trials(ax,trial_starts,trial_ends,end_clock)
    flatten_list_act = list(chain.from_iterable(acts_starts))
    flatten_list_act_end = list(chain.from_iterable(acts_ends))
    drawaction_starts(ax,flatten_list_act,flatten_list_act_end,end_clock)
    draw_events(ax,events,end_clock)

    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.tick_params(axis='x', length=10)
    ax.set_yticks([])  # turn off the yticks

    _, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(-15, xmax)
    ax.set_ylim(ymin-25, ymax+25) # make room for the legend
    ax.text(xmax, -5, "s", ha='right', va='top', size=14)
    fig.legend(ncol=5, loc='upper left')
    fig.tight_layout()

    # fig_sc,ax_sc = plt.subplots()
    # draw_scores(ax_sc,trial_scores,timestep_scores,trial_starts,trial_ends,tmstps_ends,fb_rtv)
    # ax_sc.set_title("Participant " + str(participant) + " scores")
    # ax_sc.set_xlabel("Timestamps")
    # ax_sc.set_ylabel("Score / feedback")
    # ax_sc.grid()
    # plt.show()

    return fig