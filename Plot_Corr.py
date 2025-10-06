import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from get_data_functions import *
from assemble_data_functions import *
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

numeric_pooled_arpc2ko_df = pooled_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_wt_df.select_dtypes(include=np.number)

save_path = './figures/corr_plots'

A = numeric_pooled_arpc2ko_df
B = numeric_pooled_arpc2ko_df
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    plt.plot(pooled_wt_df[column1],pooled_wt_df[column2],'.',color='tab:blue',label='WT')
    plt.plot(pooled_arpc2ko_df[column1],pooled_arpc2ko_df[column2],'.', color='tab:orange',label='ARPC2KO')

    sns.regplot(data=pooled_wt_df, x=column1, y=column2, scatter=False, ci=95, color="tab:blue")
    sns.regplot(data=pooled_arpc2ko_df, x=column1, y=column2, scatter=False, ci=95, color="tab:orange")

    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.png'.format(save_path, column1, column2),bbox_inches='tight')
    plt.clf()

    # r, pval = pearsonr(pooled_arpc2ko_df[param1],pooled_arpc2ko_df[param2])
    # print(f"Pearson r = {r:.3f}, p = {pval:.3e}")

    # r, pval = pearsonr(pooled_wt_df[param1],pooled_wt_df[param2])
    # print(f"Pearson r = {r:.3f}, p = {pval:.3e}")