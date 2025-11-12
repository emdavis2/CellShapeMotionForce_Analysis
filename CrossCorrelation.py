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
from cross_corr_functions import *

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
all_cells = wt_cells + arpc2ko_cells

data = []
lengths = []
for ind in range(len(all_cells)):
    avg_track_mag = all_cells[ind]['avg_trac_mag']
    lengths.append(len(avg_track_mag))
    data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

Data = np.vstack(data)

#set up HMM
n_states = 2
model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
model.fit(Data, lengths)

#get fitted model outputs
transmat = model.transmat_
stds = np.sqrt(model.covars_.flatten())
means = model.means_.flatten()
 # Empirical weights from predicted state sequence
X = Data
X = X.ravel()
state_seq = model.predict(X.reshape(-1, 1))
weights = np.bincount(state_seq, minlength=n_states) / len(X)

#set up data into type to predict cell states
data_arpc2ko = []
lengths_arpc2ko = []
for ind in range(len(arpc2ko_cells)):
    avg_track_mag = arpc2ko_cells[ind]['avg_trac_mag']
    lengths_arpc2ko.append(len(avg_track_mag))
    data_arpc2ko.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))
    

data_wt = []
lengths_wt = []
for ind in range(len(wt_cells)):
    avg_track_mag = wt_cells[ind]['avg_trac_mag']
    lengths_wt.append(len(avg_track_mag))
    data_wt.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))


#Predict ARPC2KO states
cell_states_arpc2ko, state_probs_arpc2ko = get_states_and_probs(lengths_arpc2ko, model, np.vstack(data_arpc2ko))

#Predict WT states
cell_states_wt, state_probs_wt = get_states_and_probs(lengths_wt, model, np.vstack(data_wt))

#Get state segment metrics
arpc2ko_cells_states, arpc2ko_state0_df, arpc2ko_state1_df, arpc2ko_state0_mean_summary_df, arpc2ko_state1_mean_summary_df, arpc2ko_state0_median_summary_df, arpc2ko_state1_median_summary_df = calculate_segment_metrics(arpc2ko_cells, cell_states_arpc2ko)
wt_cells_states, wt_state0_df, wt_state1_df, wt_state0_mean_summary_df, wt_state1_mean_summary_df, wt_state0_median_summary_df, wt_state1_median_summary_df = calculate_segment_metrics(wt_cells, cell_states_wt)


feature1 = 'eccentricity'
feature2 = 'avg_trac_mag'
lags, neg_corr_arpc2ko, pos_corr_arpc2ko, else_corr_arpc2ko, pos_corr_arpc2ko_cells, neg_corr_arpc2ko_cells, else_corr_arpc2ko_cells, neg_corr_wt, pos_corr_wt, else_corr_wt, pos_corr_wt_cells, neg_corr_wt_cells, else_corr_wt_cells, predictedstates_negcorr_arpc2ko, predictedstates_else_arpc2ko, predictedstates_negcorr_wt, predictedstates_else_wt = apply_cross_corr(feature1, feature2, arpc2ko_cells_states, wt_cells_states)

pooled_poscorr_arpc2ko = pd.concat(pos_corr_arpc2ko_cells, ignore_index=True)
pooled_negcorr_arpc2ko = pd.concat(neg_corr_arpc2ko_cells, ignore_index=True)
pooled_else_arpc2ko = pd.concat(else_corr_arpc2ko_cells, ignore_index=True)
pooled_poscorr_wt = pd.concat(pos_corr_wt_cells, ignore_index=True)
pooled_negcorr_wt = pd.concat(neg_corr_wt_cells, ignore_index=True)
pooled_else_wt = pd.concat(else_corr_wt_cells, ignore_index=True)

save_path = './figures/cross_corr_{}_{}'.format(feature1, feature2)

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

for arr in neg_corr_wt:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Strong negative cross correlations in WT for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_neg_wt.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

for arr in pos_corr_wt:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Strong positive cross correlations in WT for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_pos_wt.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

for arr in neg_corr_wt:
    plt.plot(lags,arr)
for arr in pos_corr_wt:
    plt.plot(lags,arr)
for arr in else_corr_wt:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Cross correlations in WT for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_all_wt.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

for arr in neg_corr_arpc2ko:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Strong negative cross correlations in ARPC2KO for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_neg_arpc2ko.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

for arr in pos_corr_arpc2ko:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Strong positive cross correlations in ARPC2KO for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_pos_arpc2ko.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

for arr in neg_corr_arpc2ko:
    plt.plot(lags,arr)
for arr in pos_corr_arpc2ko:
    plt.plot(lags,arr)
for arr in else_corr_arpc2ko:
    plt.plot(lags,arr)
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Cross correlations in ARPC2KO for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_all_arpc2ko.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

all_corr_arpc2ko = neg_corr_arpc2ko + pos_corr_arpc2ko + else_corr_arpc2ko
all_corr_wt = neg_corr_wt + pos_corr_wt + else_corr_wt

plt.plot(np.average(all_corr_arpc2ko,axis=0))
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Cross correlations in ARPC2KO for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_avg_arpc2ko.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

plt.plot(np.average(all_corr_wt,axis=0))
plt.xlabel('lags (15 min)')
plt.ylabel('correlation')
plt.title('Cross correlations in WT for {} and {}'.format(feature1,feature2))
plt.savefig('{}/{}_{}_crosscorr_avg_wt.png'.format(save_path, feature1, feature2),bbox_inches='tight')
plt.clf()

#clears out sentinel file if it exists
open('{}/strong_neg_wt_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_neg_wt = open('{}/strong_neg_wt_names.txt'.format(save_path),'w')
file_lines_wt = []

#clears out sentinel file if it exists
open('{}/strong_neg_arpc2ko_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_neg_arpc2ko = open('{}/strong_neg_arpc2ko_names.txt'.format(save_path),'w')
file_lines_arpc2ko = []

#clears out sentinel file if it exists
open('{}/strong_pos_wt_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_pos_wt = open('{}/strong_pos_wt_names.txt'.format(save_path),'w')
file_lines_pos_wt = []

#clears out sentinel file if it exists
open('{}/strong_pos_arpc2ko_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_pos_arpc2ko = open('{}/strong_pos_arpc2ko_names.txt'.format(save_path),'w')
file_lines_pos_arpc2ko = []

for df in neg_corr_arpc2ko_cells:
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    file_lines_arpc2ko.append(name + '\n')

for df in neg_corr_wt_cells:
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    file_lines_wt.append(name + '\n')

for df in pos_corr_arpc2ko_cells:
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    file_lines_pos_arpc2ko.append(name + '\n')

for df in pos_corr_wt_cells:
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    file_lines_pos_wt.append(name + '\n')

#write lines to text file 
names_neg_wt.writelines(file_lines_wt)
names_neg_wt.close() 

names_neg_arpc2ko.writelines(file_lines_arpc2ko)
names_neg_arpc2ko.close() 

names_pos_wt.writelines(file_lines_pos_wt)
names_pos_wt.close() 

names_pos_arpc2ko.writelines(file_lines_pos_arpc2ko)
names_pos_arpc2ko.close() 