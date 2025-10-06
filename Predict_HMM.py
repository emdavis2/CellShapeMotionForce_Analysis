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
arpc2ko_state0_df, arpc2ko_state1_df, arpc2ko_state0_summary_df, arpc2ko_state1_summary_df = calculate_segment_metrics(arpc2ko_cells, cell_states_arpc2ko)
wt_state0_df, wt_state1_df, wt_state0_summary_df, wt_state1_summary_df = calculate_segment_metrics(wt_cells, cell_states_wt)

data_bp = pd.concat([arpc2ko_state0_df,arpc2ko_state1_df,wt_state0_df,wt_state1_df]).reset_index()
save_path = './figures/stripplots'

numeric_arpc2ko_state0_df = arpc2ko_state0_df.select_dtypes(include=np.number)
numeric_arpc2ko_state1_df = arpc2ko_state1_df.select_dtypes(include=np.number)
numeric_wt_state0_df = wt_state0_df.select_dtypes(include=np.number)
numeric_wt_state1_df = wt_state1_df.select_dtypes(include=np.number)

for column in numeric_arpc2ko_state0_df.columns:
    print(column)
    stripplot_hmm(column, data_bp, save_path, '{}_stripplot'.format(column)) 