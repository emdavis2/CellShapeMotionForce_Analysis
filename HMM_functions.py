import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def get_states_and_probs(lengths, model, data):
    start = 0
    cell_states = []
    state_probs = []

    for length in lengths:
        end = start + length
        states = model.predict(data[start:end])
        probs = model.predict_proba(data[start:end])
        cell_states.append(states)
        state_probs.append(probs)
        start = end
        
    return cell_states, state_probs

#####################################################################################################################################################################################

#input is the list of dataframes (arpc2ko_cells or wt_cells) from combine_data() function in assemble_data_functions.py and the list of states from get_states_and _probs() function
#output is two dataframes of each state that has all tracks concatenated and two other dataframes of each state that has the means of the metrtcs for each track
def calculate_segment_metrics(cell_df_list, cell_states_list):
    pixel_size = 0.645 #microns

    updated_df_list = []
    for track in range(len(cell_df_list)):
        track_df = cell_df_list[track]
        states_track = cell_states_list[track]
        track_df['state'] = states_track
        updated_df_list.append(track_df)
        
      
        
    for track_df in updated_df_list:
        track_df['segment_speed'] = np.nan
        track_df['segment_DT'] = np.nan
    
        # Detect contiguous state runs
        state_change = (track_df['state'] != track_df['state'].shift()).cumsum()
    
        # Group by state runs
        for _, seg_idx in track_df.groupby(state_change).groups.items():
            segment = track_df.loc[seg_idx]
    
            if len(segment) < 2:
                continue  # Skip too-short segments
    
            # Compute stepwise distances
            x0, y0 = segment.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
            x1, y1 = segment.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
            net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
            
            dx = segment['approximate-medoidx'].diff().values[1:]
            dy = segment['approximate-medoidy'].diff().values[1:]
            step_lengths = np.sqrt(dx**2 + dy**2)
            path_length = np.sum(step_lengths) * pixel_size
            duration = (len(segment) - 1) * 15 #minutes because sampled every third frame with 5 minutes between frames
            speed = path_length / duration if duration > 0 else np.nan
            
            DT = net_displacement/path_length
    
            track_df.loc[seg_idx, 'segment_speed'] = speed
            track_df.loc[seg_idx, 'segment_DT'] = DT
            track_df.loc[seg_idx, 'segment_duration'] = duration
        
        
    final_df = pd.concat(updated_df_list, ignore_index=True)

    state0_df_final = final_df[final_df['state'] == 0].copy()
    state1_df_final = final_df[final_df['state'] == 1].copy()

    type = final_df['type'].iloc[0]
    state0_df_final['state_name'] = ['state 0 {}'.format(type)] * len(state0_df_final)
    state1_df_final['state_name'] = ['state 1 {}'.format(type)] * len(state1_df_final)

    state0_summary_df = []
    state1_summary_df = []
    for track_df in updated_df_list:
        state0_df = track_df[track_df['state'] == 0]
        state1_df = track_df[track_df['state'] == 1]
        
        if len(state0_df) > 0:
            means = state0_df.mean(numeric_only = True)
            
            means.loc['track_name']=(state0_df['experiment'].iloc[0] + '_movie' + str(int(state0_df['movie'].iloc[0])) + '_track' + str(int(state0_df['track_id'].iloc[0])))
            state0_summary_df.append(means)
            
        if len(state1_df) > 0:
            means = state1_df.mean(numeric_only = True)
            
            means.loc['track_name']=(state1_df['experiment'].iloc[0] + '_movie' + str(int(state1_df['movie'].iloc[0])) + '_track' + str(int(state1_df['track_id'].iloc[0])))
            state1_summary_df.append(means)
    
            
    state0_summary_df = pd.concat(state0_summary_df,axis=1).T
    state1_summary_df = pd.concat(state1_summary_df,axis=1).T
    
    state0_summary_df['state_name'] = ['state 0 {}'.format(type)] * len(state0_summary_df)
    state1_summary_df['state_name'] = ['state 1 {}'.format(type)] * len(state1_summary_df)

    return state0_df_final, state1_df_final, state0_summary_df, state1_summary_df

#####################################################################################################################################################################################

def stripplot_hmm(param, data_bp, save_path, plot_name):
    sns.stripplot(data=data_bp, x='state_name', y=param, hue='dishnum',alpha=0.7,palette='deep')
    sns.pointplot(data=data_bp, x="state_name", y=param,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
    sns.pointplot(data=data_bp[data_bp['dishnum']==1], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[0],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    sns.pointplot(data=data_bp[data_bp['dishnum']==2], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[1],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    sns.pointplot(data=data_bp[data_bp['dishnum']==3], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[2],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    sns.pointplot(data=data_bp[data_bp['dishnum']==4], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[3],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    sns.pointplot(data=data_bp[data_bp['dishnum']==5], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[4],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    sns.pointplot(data=data_bp[data_bp['dishnum']==6], x="state_name", y=param,linestyle="none",marker="*",color=sns.color_palette('deep')[5],markersize=20,markeredgewidth=1,errorbar=None,markeredgecolor='.5')
    
    plt.xlabel("Type")
    plt.ylabel("{}".format(param))
    plt.xticks(rotation=90)


    plt.savefig('{}/{}.png'.format(save_path, plot_name),bbox_inches='tight')
    plt.clf()