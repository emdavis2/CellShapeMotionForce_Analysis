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

pixel_size = 0.645 #microns

for df in wt_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 30 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    df['speed'] = [speed] * len(df)
    df['DT'] = [DT] * len(df)

for df in arpc2ko_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 30 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    df['speed'] = [speed] * len(df)
    df['DT'] = [DT] * len(df)


    
numeric_pooled_arpc2ko_df = pooled_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_wt_df.select_dtypes(include=np.number)

color_param = 'avg_trac_mag'

# save_path = './figures/corr_plots_wt_only_pertrack_byAvgTracMag'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

# A = numeric_pooled_wt_df
# B = numeric_pooled_wt_df
# unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

# cmin = min(df[color_param].min() for df in wt_cells)
# cmax = max(df[color_param].max() for df in wt_cells)

# for pair in unique_pairs:
#     column1 = pair[0]
#     column2 = pair[1]
#     for df in wt_cells:
#         plt.scatter(df[column1],df[column2],c=df[color_param],cmap='viridis', vmin=cmin, vmax=cmax,alpha=0.25)
#     # plt.plot(pooled_arpc2ko_df[column1],pooled_arpc2ko_df[column2],'.', color='tab:orange',label='ARPC2KO')

#     # sns.regplot(data=pooled_wt_df, x=column1, y=column2, scatter=False, ci=95, color="tab:blue")
#     # sns.regplot(data=pooled_arpc2ko_df, x=column1, y=column2, scatter=False, ci=95, color="tab:orange")

#     plt.xlabel('{}'.format(column1))
#     plt.ylabel('{}'.format(column2))
#     cbar = plt.colorbar()
#     cbar.set_label(color_param)
#     # plt.legend()

#     plt.savefig('{}/{}_{}_corrplot.png'.format(save_path, column1, column2),bbox_inches='tight')
#     plt.clf()

#     # r, pval = pearsonr(pooled_arpc2ko_df[param1],pooled_arpc2ko_df[param2])
#     # print(f"Pearson r = {r:.3f}, p = {pval:.3e}")

#     # r, pval = pearsonr(pooled_wt_df[param1],pooled_wt_df[param2])
#     # print(f"Pearson r = {r:.3f}, p = {pval:.3e}")

# save_path = './figures/corr_plots_arpc2ko_only_pertrack'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

# A = numeric_pooled_arpc2ko_df
# B = numeric_pooled_arpc2ko_df
# unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

# cmin = min(df[color_param].min() for df in arpc2ko_cells)
# cmax = max(df[color_param].max() for df in arpc2ko_cells)

# for pair in unique_pairs:
#     column1 = pair[0]
#     column2 = pair[1]
#     for df in arpc2ko_cells:
#         plt.scatter(df[column1],df[column2],c=df[color_param],cmap='viridis', vmin=cmin, vmax=cmax,alpha=0.25)
#         # plt.plot(df[column1],df[column2],'.-',alpha=0.25)
#     # plt.plot(pooled_arpc2ko_df[column1],pooled_arpc2ko_df[column2],'.', color='tab:orange',label='ARPC2KO')

#     # sns.regplot(data=pooled_wt_df, x=column1, y=column2, scatter=False, ci=95, color="tab:blue")
#     # sns.regplot(data=pooled_arpc2ko_df, x=column1, y=column2, scatter=False, ci=95, color="tab:orange")

#     plt.xlabel('{}'.format(column1))
#     plt.ylabel('{}'.format(column2))
#     cbar = plt.colorbar()
#     cbar.set_label(color_param)
#     # plt.legend()

#     plt.savefig('{}/{}_{}_corrplot.png'.format(save_path, column1, column2),bbox_inches='tight')
#     plt.clf()


save_path = './figures/corr_plots_pertrack_byAvgTracMag'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

A = numeric_pooled_arpc2ko_df
B = numeric_pooled_arpc2ko_df
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

cmin_arpc2ko = min(df[color_param].min() for df in arpc2ko_cells)
cmax_arpc2ko = max(df[color_param].max() for df in arpc2ko_cells)
cmin_wt = min(df[color_param].min() for df in wt_cells)
cmax_wt = max(df[color_param].max() for df in wt_cells)

cmin = min([cmin_arpc2ko, cmin_wt])
cmax = max([cmax_arpc2ko, cmax_wt])

for pair in unique_pairs:
    # Create subplots with shared x and y axes
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

    column1 = pair[0]
    column2 = pair[1]
    for df in arpc2ko_cells:
        im1 = axs[0].scatter(df[column1],df[column2],c=df[color_param],cmap='viridis', vmin=cmin, vmax=cmax,alpha=0.25)
        # axs[0].plot(df[column1],df[column2],'.-',alpha=0.25)
        axs[0].set_title('ARPC2KO')
    
    for df in wt_cells:
        im2 = axs[1].scatter(df[column1],df[column2],c=df[color_param],cmap='viridis', vmin=cmin, vmax=cmax,alpha=0.25)
        # axs[1].plot(df[column1],df[column2],'.-',alpha=0.25)
        axs[1].set_title('WT')

    for ax in axs:
        ax.set_box_aspect(1) # Sets the box aspect ratio to 1:1 (square)

    fig.supxlabel('{}'.format(column1))
    fig.supylabel('{}'.format(column2))

    # fig.suptitle('{} and {}'.format(column1, column2))

    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.8, orientation='vertical')
    cbar.set_label(color_param)

    plt.savefig('{}/{}_{}_corrplot.png'.format(save_path, column1, column2),bbox_inches='tight')
    plt.clf()
    plt.close()