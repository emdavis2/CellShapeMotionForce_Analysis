import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions import *
from plot_dist_functions import *


#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)

csv_path = data_dir + '/skeleton.csv'
skeleton_df = load_skeletondf(csv_path)

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df)

#format data to input to HMM
all_cells = arpc2ko_cells + wt_cells

obs_param = 'avg_trac_mag'

data = []
lengths = []
for ind in range(len(all_cells)):
    avg_track_mag = all_cells[ind][obs_param]
    lengths.append(len(avg_track_mag))
    data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

Data = np.vstack(data)

#set up HMM
n_states = 2
model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
model.fit(Data, lengths)

#get fitted model outputs
transmat = model.transmat_
dt = 5/60 #hours since dt is 5 min and I want to convert to rate per hour
transmat_rate = -(1/dt)*np.log(1-transmat)
stds = np.sqrt(model.covars_.flatten())
means = model.means_.flatten()
 # Empirical weights from predicted state sequence
X = Data
X = X.ravel()
state_seq = model.predict(X.reshape(-1, 1))
weights = np.bincount(state_seq, minlength=n_states) / len(X)

save_path = './figures/HMM_{}'.format(obs_param)
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

### Plot model
feature_name="{}".format(obs_param)
# Plot histogram of data
plt.figure(figsize=(10, 5))
plt.hist(X.flatten(), bins=50, density=True, alpha=0.3, color='gray', label="Observed data")
# Overlay Gaussian PDFs for each state
x_vals = np.linspace(X.min(), X.max(), 1000)
weighted_sum = np.zeros_like(x_vals)
for i, (mean, std) in enumerate(zip(means, stds)):
    y = norm.pdf(x_vals, loc=mean, scale=std)
    plt.plot(x_vals, weights[i] * y, label=f"State {i} (mean={means[i]:.2f})", linewidth=2)
    weighted_pdf = weights[i] * y
    weighted_sum += weighted_pdf
# Plot the mixture
plt.plot(x_vals, weighted_sum, 'k--', lw=2, label='Weighted mixture')
plt.title(f"HMM Emission Distributions for {feature_name}")
plt.xlabel(feature_name)
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig(save_path + '/HMM_Model.png')
plt.clf()

title="HMM Transition Probabilities"
state_labels = [f"State {i}" for i in range(n_states)]
plt.figure(figsize=(6, 5))
sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=state_labels, yticklabels=state_labels,
            square=True, cbar_kws={'label': 'P(transition)'})
plt.xlabel("To State")
plt.ylabel("From State")
plt.title(title)
plt.tight_layout()
plt.savefig(save_path+'/transition_matrix.png')
plt.clf()

title="HMM Transition Probabilities"
state_labels = [f"State {i}" for i in range(n_states)]
plt.figure(figsize=(6, 5))
sns.heatmap(transmat_rate, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=state_labels, yticklabels=state_labels,
            square=True, cbar_kws={'label': 'P(transition)'})
plt.xlabel("To State")
plt.ylabel("From State")
plt.title(title)
plt.tight_layout()
plt.savefig(save_path+'/transition_matrix_rate.png')
plt.clf()

#set up data into type to predict cell states
data_arpc2ko = []
lengths_arpc2ko = []
for ind in range(len(arpc2ko_cells)):
    avg_track_mag = arpc2ko_cells[ind][obs_param]
    lengths_arpc2ko.append(len(avg_track_mag))
    data_arpc2ko.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))
    

data_wt = []
lengths_wt = []
for ind in range(len(wt_cells)):
    avg_track_mag = wt_cells[ind][obs_param]
    lengths_wt.append(len(avg_track_mag))
    data_wt.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))


#Predict ARPC2KO states
cell_states_arpc2ko, state_probs_arpc2ko = get_states_and_probs(lengths_arpc2ko, model, np.vstack(data_arpc2ko))

#Predict WT states
cell_states_wt, state_probs_wt = get_states_and_probs(lengths_wt, model, np.vstack(data_wt))

#Get state segment metrics
arpc2ko_cells_states, arpc2ko_state0_df, arpc2ko_state1_df, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df = calculate_segment_metrics(arpc2ko_cells, cell_states_arpc2ko, state_probs_arpc2ko)
wt_cells_states, wt_state0_df, wt_state1_df, wt_state0_mean_summary_df, wt_state1_mean_summary_df, wt_state0_median_summary_df, wt_state1_median_summary_df = calculate_segment_metrics(wt_cells, cell_states_wt, state_probs_wt)

#clears out sentinel file if it exists
open('{}/transition_wt_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_trans_wt = open('{}/transition_wt_names.txt'.format(save_path),'w')
file_lines_wt = []

#clears out sentinel file if it exists
open('{}/transition_arpc2ko_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_trans_arpc2ko = open('{}/transition_arpc2ko_names.txt'.format(save_path),'w')
file_lines_arpc2ko = []

for df in arpc2ko_cells_states:
    states = df['state']
    if len(np.unique(states)) > 1:
        name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
        n_transitions = np.sum(np.diff(states) != 0)
        file_lines_arpc2ko.append(name + ' num transtions: {}'.format(n_transitions) + '\n')

for df in wt_cells_states:
    states = df['state']
    if len(np.unique(states)) > 1:
        name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
        n_transitions = np.sum(np.diff(states) != 0)
        file_lines_wt.append(name + ' num transtions: {}'.format(n_transitions) + '\n')

#write lines to text file 
names_trans_wt.writelines(file_lines_wt)
names_trans_wt.close() 

names_trans_arpc2ko.writelines(file_lines_arpc2ko)
names_trans_arpc2ko.close() 

numeric_arpc2ko_state0_df = arpc2ko_state0_df.select_dtypes(include=np.number)
numeric_arpc2ko_state1_df = arpc2ko_state1_df.select_dtypes(include=np.number)
numeric_wt_state0_df = wt_state0_df.select_dtypes(include=np.number)
numeric_wt_state1_df = wt_state1_df.select_dtypes(include=np.number)

# data_bp = pd.concat([arpc2ko_state0_df,arpc2ko_state1_df,wt_state0_df,wt_state1_df]).reset_index()
# save_path_fig = '{}/stripplots'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column))
#     plot_dist(column, arpc2ko_state0_df, arpc2ko_state1_df, 'State 0', 'State 1', save_path_fig, column + '_ARPC2KO_distribution')  
#     plot_dist(column, wt_state0_df, wt_state1_df, 'State 0', 'State 1', save_path_fig, column + '_WT_distribution') 
#     plot_dist_all(column, wt_state0_df, wt_state1_df, arpc2ko_state0_df, arpc2ko_state1_df, 'WT State 0', 'WT State 1', 'ARPC2KO State 0', 'ARPC2KO State 1', save_path_fig, column + '_distribution')


# data_bp = pd.concat([arpc2ko_state0_mean_summary_df,arpc2ko_state1_mean_summary_df,wt_state0_mean_summary_df,wt_state1_mean_summary_df]).reset_index()
# save_path_fig = '{}/stripplots_mean'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column))
#     plot_dist(column, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, 'State 0', 'State 1', save_path_fig, column + '_ARPC2KO_mean_distribution')  
#     plot_dist(column, wt_state0_mean_summary_df, wt_state1_mean_summary_df, 'State 0', 'State 1', save_path_fig, column + '_WT_mean_distribution') 
#     plot_dist_all(column, wt_state0_mean_summary_df, wt_state1_mean_summary_df, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, 'WT State 0', 'WT State 1', 'ARPC2KO State 0', 'ARPC2KO State 1', save_path_fig, column + '_mean_distribution')


# data_bp = pd.concat([arpc2ko_state0_median_summary_df,arpc2ko_state1_median_summary_df,wt_state0_median_summary_df,wt_state1_median_summary_df]).reset_index()
# save_path_fig = '{}/stripplots_median'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column)) 
#     plot_dist(column, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df, 'State 0', 'State 1', save_path_fig, column + '_ARPC2KO_median_distribution')  
#     plot_dist(column, wt_state0_median_summary_df, wt_state1_median_summary_df, 'State 0', 'State 1', save_path_fig, column + '_WT_median_distribution')
#     plot_dist_all(column, wt_state0_median_summary_df, wt_state1_median_summary_df, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df, 'WT State 0', 'WT State 1', 'ARPC2KO State 0', 'ARPC2KO State 1', save_path_fig, column + '_median_distribution')

segment_speeds_cells = []
segment_DT_cells = []
for df in wt_cells_states:
  # define segment IDs whenever state changes
  df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
  # now get one row per segment (e.g. first row of each)
  segments = df.groupby('segment_id', as_index=False).first()
  segment_speeds_cells.append(segments[['segment_speed', 'track_name', 'state_name', 'state']])
  segment_DT_cells.append(segments[['segment_DT','track_name', 'state_name', 'state']])

for df in arpc2ko_cells_states:
    # define segment IDs whenever state changes
    df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
    # now get one row per segment (e.g. first row of each)
    segments = df.groupby('segment_id', as_index=False).first()
    segment_speeds_cells.append(segments[['segment_speed', 'track_name', 'state_name', 'state']])
    segment_DT_cells.append(segments[['segment_DT','track_name', 'state_name', 'state']])

segment_speeds = pd.concat(segment_speeds_cells, ignore_index=True)
segment_DT = pd.concat(segment_DT_cells, ignore_index=True)

sns.stripplot(data=segment_speeds, x='state_name', y='segment_speed', hue='track_name',alpha=0.7,palette='deep',legend=False)
sns.pointplot(data=segment_speeds, x="state_name", y='segment_speed',linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
plt.xlabel("Type")
plt.ylabel("Segment Speed")
plt.xticks(rotation=90)
plt.savefig('{}/segment_speed_stripplot.png'.format(save_path),bbox_inches='tight')
plt.clf()

hist, bin_edges = np.histogram(segment_speeds[segment_speeds['state_name']=='state 0 WT']['segment_speed'].dropna(),bins=15)
plt.hist(segment_speeds[segment_speeds['state_name']=='state 0 WT']['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='blue',label='{}'.format('WT State 0'),density=True)
plt.hist(segment_speeds[segment_speeds['state_name']=='state 1 WT']['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='red',label='{}'.format('WT State 1'),density=True)
plt.hist(segment_speeds[segment_speeds['state_name']=='state 0 ARPC2KO']['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='green',label='{}'.format('ARPC2KO State 0'),density=True)
plt.hist(segment_speeds[segment_speeds['state_name']=='state 1 ARPC2KO']['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='orange',label='{}'.format('ARPC2KO State 1'),density=True)
# plt.title('{}'.format(feature))
plt.title('Segment Speed Distribution')
plt.xlabel('Segment Speed')
plt.ylabel('density')
plt.legend()
plt.savefig('{}/SegmentSpeed_distirbution.png'.format(save_path),bbox_inches='tight')
plt.clf()


sns.stripplot(data=segment_DT, x='state_name', y='segment_DT', hue='track_name',alpha=0.7,palette='deep',legend=False)
sns.pointplot(data=segment_DT, x="state_name", y='segment_DT',linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
plt.xlabel("Type")
plt.ylabel("Segment DT")
plt.xticks(rotation=90)
plt.savefig('{}/segment_DT_stripplot.png'.format(save_path),bbox_inches='tight')
plt.clf()

hist, bin_edges = np.histogram(segment_DT[segment_DT['state_name']=='state 0 WT']['segment_DT'].dropna(),bins=15)
plt.hist(segment_DT[segment_DT['state_name']=='state 0 WT']['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='blue',label='{}'.format('WT State 0'),density=True)
plt.hist(segment_DT[segment_DT['state_name']=='state 1 WT']['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='red',label='{}'.format('WT State 1'),density=True)
plt.hist(segment_DT[segment_DT['state_name']=='state 0 ARPC2KO']['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='green',label='{}'.format('ARPC2KO State 0'),density=True)
plt.hist(segment_DT[segment_DT['state_name']=='state 1 ARPC2KO']['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='orange',label='{}'.format('ARPC2KO State 1'),density=True)
# plt.title('{}'.format(feature))
plt.title('Segment DT Distribution')
plt.xlabel('Segment DT')
plt.ylabel('density')
plt.legend()
plt.savefig('{}/SegmentDT_distirbution.png'.format(save_path),bbox_inches='tight')
plt.clf()