import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *


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
    "median_protrusion_widths": parse_list,
    "mean_protrusion_widths": parse_list
})


arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df, protrusion_df)

A = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']
B = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

save_path = './figures/spearman_plots_wt'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    rho_vals = []
    p_vals = []
    for df in wt_cells:
        rho, pval = spearmanr(df[column1],df[column2])
        if pval < 0.05:
          rho_vals.append(rho)
          p_vals.append(pval)

    sns.violinplot(x=rho_vals, inner=None, legend=False)
    sns.swarmplot(x=rho_vals,legend=True, hue=p_vals,size=3)
    plt.title("{} and {}".format(column1, column2))

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()


save_path = './figures/spearman_plots_arpc2ko'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    rho_vals = []
    p_vals = []
    for df in arpc2ko_cells:
        rho, pval = spearmanr(df[column1],df[column2])
        if pval < 0.05:
          rho_vals.append(rho)
          p_vals.append(pval)

    sns.violinplot(x=rho_vals, inner=None, legend=False)
    sns.swarmplot(x=rho_vals,legend=True, hue=p_vals,size=3)
    plt.title("{} and {}".format(column1, column2))

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()