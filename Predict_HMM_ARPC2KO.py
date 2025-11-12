import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit 
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions import *
from plot_dist_functions import *
from stat_tests import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)

csv_path = data_dir + '/skeleton.csv'
skeleton_df = load_skeletondf(csv_path)

protr_csv_path = data_dir + '/protrusion.csv'
protrusion_df = pd.read_csv(protr_csv_path, converters={
   "protrusion_lengths": parse_list,
    "median_protrusion_widths": parse_list,
    "mean_protrusion_widths": parse_list
})

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df, protrusion_df)

#format data to input to HMM
all_cells = arpc2ko_cells

obs_param = 'avg_trac_mag'

save_path = './figures/HMM_{}_ARPC2KO'.format(obs_param)
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

data = []
lengths = []
for ind in range(len(all_cells)):
    avg_track_mag = all_cells[ind][obs_param]
    lengths.append(len(avg_track_mag))
    data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

Data = np.vstack(data)

#Fit sum of two Gaussians functions to avg trac mag histogram
hist,bin_edges = np.histogram(Data.flatten(), bins=50, density=True)
popt, pcov = curve_fit(lambda trac, sigma1, mu1, sigma2, mu2, w1, w2: w1*((1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((trac - mu1)/(sigma1))**2)) + w2*((1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((trac - mu2)/(sigma2))**2)), np.linspace(Data.min(), Data.max(), len(hist)), hist, bounds=([1,Data.min(),1,Data.min(),0.1,0.1],[20,Data.max(),20,Data.max(),1,1]))
sigma1 = popt[0]
mu1 = popt[1]
sigma2 = popt[2]
mu2 = popt[3]
w1 = popt[4]
w2 = popt[5]
x_fitted = np.linspace(Data.min(), Data.max(), 1000)
y_fitted = w1*((1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x_fitted - mu1)/(sigma1))**2)) + w2*((1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x_fitted - mu2)/(sigma2))**2))

plt.plot(x_fitted,y_fitted, label="means {} and {}; covars {} and {}".format(np.round(mu1,2),np.round(mu2,2),np.round(sigma1**2,2),np.round(sigma2**2,2)))
plt.hist(Data.flatten(), bins=50, density=True, alpha=0.3, color='gray', label="Observed data")
plt.xlabel('{}'.format(obs_param))
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig(save_path+'/fit_to_{}.pdf'.format(obs_param),format='pdf')
plt.clf()

mus = [mu1, mu2]
sigmas = [sigma1, sigma2]
low_ind = np.argmin(mus)
high_ind = np.argmax(mus)
means_model = np.array([[mus[low_ind]],[mus[high_ind]]])
covars_model = np.array([[[sigmas[low_ind]**2]], [[sigmas[high_ind]**2]]])

#set up HMM
n_states = 2
model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42,means_prior=means_model, covars_prior=covars_model,init_params='st',params='st')
model.means_ = means_model
model.covars_ = covars_model
model.fit(Data, lengths)

#get fitted model outputs
transmat = model.transmat_
dt = 15/60 #hours since dt is 15 min and I want to convert to rate per hour
transmat_rate = -(1/dt)*np.log(1-transmat)
stds = np.sqrt(model.covars_.flatten())
means = model.means_.flatten()
 # Empirical weights from predicted state sequence
X = Data
X = X.ravel()
state_seq = model.predict(X.reshape(-1, 1))
weights = np.bincount(state_seq, minlength=n_states) / len(X)
states = ['low force', 'high force']

colors_for_state = [(242/256,140/256,40/256), (11/256,218/256,81/256)]
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
    plt.plot(x_vals, weights[i] * y, color=colors_for_state[i], label=f"State {states[i]} (mean={means[i]:.2f})", linewidth=2)
    weighted_pdf = weights[i] * y
    weighted_sum += weighted_pdf
# Plot the mixture
plt.plot(x_vals, weighted_sum, 'k--', lw=2, label='Weighted mixture')
plt.title(f"HMM Emission Distributions for {feature_name}")
plt.xlabel(feature_name)
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig(save_path + '/HMM_Model.pdf',format='pdf')
plt.clf()

title="HMM Transition Probabilities"
state_labels = [f"State {states[i]}" for i in range(n_states)]
plt.figure(figsize=(6, 5))
sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=state_labels, yticklabels=state_labels,
            square=True, cbar_kws={'label': 'P(transition)'})
plt.xlabel("To State")
plt.ylabel("From State")
plt.title(title)
plt.tight_layout()
plt.savefig(save_path+'/transition_matrix.pdf',format='pdf')
plt.clf()

title="HMM Transition Rates"
state_labels = [f"State {states[i]}" for i in range(n_states)]
plt.figure(figsize=(6, 5))
sns.heatmap(transmat_rate, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=state_labels, yticklabels=state_labels,
            square=True, cbar_kws={'label': 'P(transition)'})
plt.xlabel("To State")
plt.ylabel("From State")
plt.title(title)
plt.tight_layout()
plt.savefig(save_path+'/transition_matrix_rate.pdf',format='pdf')
plt.clf()

#set up data into type to predict cell states

data_arpc2ko = []
lengths_arpc2ko = []
for ind in range(len(arpc2ko_cells)):
    avg_track_mag = arpc2ko_cells[ind][obs_param]
    lengths_arpc2ko.append(len(avg_track_mag))
    data_arpc2ko.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))


#Predict ARPC2KO states
cell_states_arpc2ko, state_probs_arpc2ko = get_states_and_probs(lengths_arpc2ko, model, np.vstack(data_arpc2ko))

#Get state segment metrics
arpc2ko_cells_states, arpc2ko_state0_df, arpc2ko_state1_df, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df = calculate_segment_metrics(arpc2ko_cells, cell_states_arpc2ko, state_probs_arpc2ko)

#clears out sentinel file if it exists
open('{}/transition_arpc2ko_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_trans_arpc2ko = open('{}/transition_arpc2ko_names.txt'.format(save_path),'w')
file_lines_arpc2ko = []

low_to_high = 0
high_to_low = 0
no_transition = 0
num_transitions = []
for df in arpc2ko_cells_states:
    states = df['state']
    if len(np.unique(states)) > 1:
        name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
        n_transitions = np.sum(np.diff(states) != 0)
        num_transitions.append(n_transitions)
        high_to_low += np.count_nonzero(np.diff(states) == -1)
        low_to_high += np.count_nonzero(np.diff(states) == 1)
        file_lines_arpc2ko.append(name + ' num transtions: {}'.format(n_transitions) + '\n')
    else:
       no_transition += 1

transition_cells = len(num_transitions)

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['No state transition', 'State transition']
counts = [no_transition, transition_cells]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of cell transitions')
plt.savefig('{}/ARPC2KO_transitions.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
values, counts = np.unique(num_transitions, return_counts=True)
categories = values.astype(object)
categories = np.array([str(x) for x in categories], dtype=object)
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of transitions for cells that transition')
plt.savefig('{}/ARPC2KO_count_in_transitions.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['Low to high', 'High to low']
counts = [low_to_high, high_to_low]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('State transition types')
plt.savefig('{}/ARPC2KO_direction_of_transition.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#write lines to text file 
names_trans_arpc2ko.writelines(file_lines_arpc2ko)
names_trans_arpc2ko.close() 

numeric_arpc2ko_state0_df = arpc2ko_state0_df.select_dtypes(include=np.number)
numeric_arpc2ko_state1_df = arpc2ko_state1_df.select_dtypes(include=np.number)


# data_bp = pd.concat([arpc2ko_state0_df,arpc2ko_state1_df]).reset_index()
# save_path_fig = '{}/stripplots'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column))
#     plot_dist(column, arpc2ko_state0_df, arpc2ko_state1_df, 'Low Force', 'High Force', save_path_fig, column + '_ARPC2KO_distribution') 


# data_bp = pd.concat([arpc2ko_state0_mean_summary_df,arpc2ko_state1_mean_summary_df]).reset_index()
# save_path_fig = '{}/stripplots_mean'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column)) 
#     plot_dist(column, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, 'Low Force', 'High Force', save_path_fig, column + '_ARPC2KO_mean_distribution') 


# data_bp = pd.concat([arpc2ko_state0_median_summary_df,arpc2ko_state1_median_summary_df]).reset_index()
# save_path_fig = '{}/stripplots_median'.format(save_path)
# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path_fig):
#   os.mkdir(save_path_fig)
# for column in numeric_arpc2ko_state0_df.columns:
#     print(column)
#     stripplot_hmm(column, data_bp, save_path_fig, '{}_stripplot'.format(column)) 
#     plot_dist(column, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df, 'Low Force', 'High Force', save_path_fig, column + '_ARPC2KO_median_distribution')

states = {0: 'low force', 1: 'high force'}

pixel_size = 0.645 #microns

segment_speeds_cells = []
segment_DT_cells = []
segment_df_cells = []
for df in arpc2ko_cells_states:
  # define segment IDs whenever state changes
  df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
  # now get one row per segment (e.g. first row of each)
  segments = df.groupby('segment_id', as_index=False).first()
  segment_speeds_cells.append(segments[['segment_speed', 'track_name', 'state_name', 'state']])
  segment_DT_cells.append(segments[['segment_DT','track_name', 'state_name', 'state']])

  column = "protrusion_lengths"
  df.loc[df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
  df[column] = df[column].apply(np.median) * pixel_size
  column = "median_protrusion_widths"
  df.loc[df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
  df[column] = df[column].apply(np.median) * pixel_size
  segment_df = df.groupby('segment_id').median()
  segment_df['track_name'] = [df['track_name'].iloc[0]] * len(segment_df)
  segment_df['state_name'] = segment_df['state'].astype(int).map(states)
  segment_df_cells.append(segment_df)

segment_speeds = pd.concat(segment_speeds_cells, ignore_index=True)
segment_DT = pd.concat(segment_DT_cells, ignore_index=True)
segment_df_all = pd.concat(segment_df_cells, ignore_index=True)

# sns.stripplot(data=segment_speeds, x='state_name', y='segment_speed', hue='track_name',alpha=0.7,palette='deep',legend=False)
# sns.pointplot(data=segment_speeds, x="state_name", y='segment_speed',linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
# plt.xlabel("Type")
# plt.ylabel("Segment Speed")
# plt.xticks(rotation=90)
# plt.savefig('{}/segment_speed_stripplot.png'.format(save_path),bbox_inches='tight')
# plt.clf()

# hist, bin_edges = np.histogram(segment_speeds[segment_speeds['state']==0]['segment_speed'].dropna(),bins=15)
# plt.hist(segment_speeds[segment_speeds['state']==0]['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='blue',label='{}'.format('Low Force'),density=True)
# plt.hist(segment_speeds[segment_speeds['state']==1]['segment_speed'].dropna(),bins=bin_edges,histtype='step',color='red',label='{}'.format('High Force'),density=True)
# # plt.title('{}'.format(feature))
# plt.title('Segment Speed Distribution')
# plt.xlabel('Segment Speed')
# plt.ylabel('density')
# plt.legend()
# plt.savefig('{}/SegmentSpeed_distirbution.png'.format(save_path),bbox_inches='tight')
# plt.clf()


# sns.stripplot(data=segment_DT, x='state_name', y='segment_DT', hue='track_name',alpha=0.7,palette='deep',legend=False)
# sns.pointplot(data=segment_DT, x="state_name", y='segment_DT',linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
# plt.xlabel("Type")
# plt.ylabel("Segment DT")
# plt.xticks(rotation=90)
# plt.savefig('{}/segment_DT_stripplot.png'.format(save_path),bbox_inches='tight')
# plt.clf()

# hist, bin_edges = np.histogram(segment_DT[segment_DT['state']==0]['segment_DT'].dropna(),bins=15)
# plt.hist(segment_DT[segment_DT['state']==0]['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='blue',label='{}'.format('Low Force'),density=True)
# plt.hist(segment_DT[segment_DT['state']==1]['segment_DT'].dropna(),bins=bin_edges,histtype='step',color='red',label='{}'.format('High Force'),density=True)
# # plt.title('{}'.format(feature))
# plt.title('Segment DT Distribution')
# plt.xlabel('Segment DT')
# plt.ylabel('density')
# plt.legend()
# plt.savefig('{}/SegmentDT_distirbution.png'.format(save_path),bbox_inches='tight')
# plt.clf()


####Median Plots###
save_path = './figures/HMM_{}_ARPC2KO/median_segment_stripplots/pdf_vp'.format(obs_param)
numeric_segment_df_all = segment_df_all.select_dtypes(include=np.number)

if not os.path.exists(save_path):
  os.mkdir(save_path)

colors_for_state_plot = [(242/256,140/256,40/256),(11/256,218/256,81/256)]

params = ['segment_speed','area','avg_trac_mag','segment_DT','segment_duration','eccentricity','solidity','dip_ratio','turning_angle','step_length',"protrusion_lengths","median_protrusion_widths",'number_protrusions']
for column in params: #numeric_segment_df_all.columns:
    # sns.stripplot(data=segment_df_all, x='state_name', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
    # sns.pointplot(data=segment_df_all, x="state_name", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
    sns.violinplot(data=segment_df_all, x='state_name', y=column, hue="state_name", order=['low force', 'high force'], palette=colors_for_state_plot, alpha=0.8, inner=None, legend=False)
    sns.swarmplot(data=segment_df_all, x='state_name', y=column, hue='track_name', palette='deep',legend=False)
    per_cell_state = (
    segment_df_all
    .groupby(['track_name', 'state_name'], observed=True)[column]
    .median()
    .reset_index()
    .rename(columns={column:'value'})
    )
    obs_stat, p, perm_stat = paired_preserving_permutation_test(per_cell_state, 'low force', 'high force', n_perm=10000, rng_seed=0)
    # data1 = segment_df_all[segment_df_all['state_name'] == 'high force']['{}'.format(column)]
    # data2 = segment_df_all[segment_df_all['state_name'] == 'low force']['{}'.format(column)]
    # U1, p = mannwhitneyu(data1, data2)
    plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig('{}/{}_stripplot.pdf'.format(save_path,column),bbox_inches='tight',format='pdf')
    plt.clf()

# ####Corr plots###
# save_path = './figures/HMM_{}_ARPC2KO/corr_plots'.format(obs_param)

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

# A = numeric_arpc2ko_state0_df
# B = numeric_arpc2ko_state0_df
# unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

# for pair in unique_pairs:
#     column1 = pair[0]
#     column2 = pair[1]
#     plt.scatter(arpc2ko_state0_df[column1],arpc2ko_state0_df[column2],color='tab:blue',alpha=0.25,label='low force')
#     plt.scatter(arpc2ko_state1_df[column1],arpc2ko_state1_df[column2],color='tab:red',alpha=0.25,label='high force')

#     plt.xlabel('{}'.format(column1))
#     plt.ylabel('{}'.format(column2))
#     plt.legend()

#     plt.savefig('{}/{}_{}_corrplot.png'.format(save_path, column1, column2),bbox_inches='tight')
#     plt.clf()