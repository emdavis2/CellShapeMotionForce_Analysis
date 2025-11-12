import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit 
from hmmlearn.hmm import GaussianHMM

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

protr_csv_path = data_dir + '/protrusion.csv'
protrusion_df = pd.read_csv(protr_csv_path, index_col=False, converters={
    "median_protrusion_widths": parse_list,
    "mean_protrusion_widths": parse_list
})

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df, protrusion_df)

#format data to input to HMM
all_cells = wt_cells

obs_param = 'avg_trac_mag'

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

#set up data into type to predict cell states

data_wt = []
lengths_wt = []
for ind in range(len(wt_cells)):
    avg_track_mag = wt_cells[ind][obs_param]
    lengths_wt.append(len(avg_track_mag))
    data_wt.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))

#Predict WT states
cell_states_wt, state_probs_wt = get_states_and_probs(lengths_wt, model, np.vstack(data_wt))

#Get state segment metrics
wt_cells_states, wt_state0_df, wt_state1_df, wt_state0_mean_summary_df, wt_state1_mean_summary_df, wt_state0_median_summary_df, wt_state1_median_summary_df = calculate_segment_metrics(wt_cells, cell_states_wt, state_probs_wt)

pooled_wt_df = pd.concat(wt_cells_states, ignore_index=True)

save_path = './figures/2D_Hist_WT'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

A = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']
B = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]

    sns.histplot(pooled_wt_df, x=column1, y=column2, hue='state')
    plt.savefig('{}/{}_{}_histplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()