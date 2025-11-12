import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit 
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm

from skimage.io import imread, imsave, imshow

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions import *
from plot_dist_functions import *
from plot_state_functions import *

track_name = '20240906_ARPC2KO_movie45_track12'
base_path = './data/tracks/{}'.format(track_name)
save_path = './figures/HMM_avgtracmag_{}'.format(track_name)
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

save_path_movie = save_path + '/states_svg'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path_movie):
  os.mkdir(save_path_movie)

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
    plt.plot(x_vals, weights[i] * y, label=f"State {states[i]} (mean={means[i]:.2f})", linewidth=2)
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

for df in arpc2ko_cells_states:
    states = df['state']
    if len(np.unique(states)) > 1:
        name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
        n_transitions = np.sum(np.diff(states) != 0)
        file_lines_arpc2ko.append(name + ' num transtions: {}'.format(n_transitions) + '\n')


names_trans_arpc2ko.writelines(file_lines_arpc2ko)
names_trans_arpc2ko.close() 


for df in arpc2ko_cells_states:
    if track_name in df['name'].iloc[0]:
        avg_trac_mag = df['avg_trac_mag']
        states = df['state']
        probs = df['prob_state0']
        frames = df['frame'].to_numpy()
        framenum = frames-1
        plot_1d_hmm_state_trace_and_probs(avg_trac_mag, states, probs, track_name, frames, save_path)
        plot_state_track(base_path, states, framenum, save_path_movie)