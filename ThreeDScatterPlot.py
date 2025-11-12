import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *

from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

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

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df)

pixel_size = 0.645 #microns


med_arpc2ko_cells = []
for df in arpc2ko_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    med_df = df.median(numeric_only=True)
    med_df['speed'] = speed
    med_df['DT'] = DT
    
    med_arpc2ko_cells.append(med_df)

med_wt_cells = []
for df in wt_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    med_df = df.median(numeric_only=True)
    med_df['speed'] = speed
    med_df['DT'] = DT
    
    med_wt_cells.append(med_df)

pooled_med_arpc2ko_df = pd.concat(med_arpc2ko_cells, axis=1, ignore_index=False).T
pooled_med_wt_df = pd.concat(med_wt_cells, axis=1, ignore_index=False).T

pooled_med_arpc2ko_df['type'] = ['ARPC2KO']*len(pooled_med_arpc2ko_df)
pooled_med_wt_df['type'] = ['WT']*len(pooled_med_wt_df)

numeric_pooled_arpc2ko_df = pooled_med_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_med_wt_df.select_dtypes(include=np.number)


color_param = 'avg_trac_mag'

plt.figure(1)

save_path = './figures/3d_corr_plots_wt_only_median_byavgtracmag'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

cmin = pooled_med_wt_df[color_param].min()
cmax = pooled_med_wt_df[color_param].max()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

# Example data (replace these with your arrays)
x = pooled_med_wt_df['area']
y = pooled_med_wt_df['eccentricity']
z = pooled_med_wt_df['solidity']

ax.scatter(x, y, z, s=50, c=pooled_med_wt_df[color_param],cmap='viridis', vmin=cmin, vmax=cmax)  # s=marker size, c=color

ax.set_xlabel("Area")
ax.set_ylabel("Eccentricity")
ax.set_zlabel("Solidity")
ax.set_title("Shape and traction force WT")

plt.show()


# plt.figure(2)

# save_path = './figures/3d_corr_plots_arpc2ko_only_median_byavgtracmag'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

# cmin = pooled_med_wt_df[color_param].min()
# cmax = pooled_med_wt_df[color_param].max()

# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(111, projection='3d')

# # Example data (replace these with your arrays)
# x = pooled_med_arpc2ko_df['area']
# y = pooled_med_arpc2ko_df['eccentricity']
# z = pooled_med_arpc2ko_df['solidity']

# ax.scatter(x, y, z, s=50, c=pooled_med_arpc2ko_df[color_param],cmap='viridis', vmin=cmin, vmax=cmax)  # s=marker size, c=color

# ax.set_xlabel("Area")
# ax.set_ylabel("Eccentricity")
# ax.set_zlabel("Solidity")
# ax.set_title("Shape and traction force Arpc2KO")

# plt.show()