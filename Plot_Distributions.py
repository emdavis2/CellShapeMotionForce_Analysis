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

save_path = './figures/histogramsII'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

for column in numeric_pooled_arpc2ko_df.columns:
    plot_dist(column, numeric_pooled_arpc2ko_df, numeric_pooled_wt_df, 'ARPC2KO', 'WT', save_path, column + ' distribution') 

pixel_size = 0.645 #microns
motion_df = {'speed':[], 'DT':[], 'type':[]}

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

    motion_df['speed'].append(speed)
    motion_df['DT'].append(DT)
    motion_df['type'].append(df['type'].iloc[0])
    
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

    motion_df['speed'].append(speed)
    motion_df['DT'].append(DT)
    motion_df['type'].append(df['type'].iloc[0])

motion_df = pd.DataFrame(motion_df)

wt_motion_df = motion_df[motion_df['type']=='WT']
arpc2ko_motion_df = motion_df[motion_df['type']=='ARPC2KO']

feature = 'speed'
hist, bin_edges = np.histogram(wt_motion_df[feature].dropna(),bins=15)
plt.hist(wt_motion_df[feature].dropna(),bins=bin_edges,histtype='step',color='blue',label='WT',density=True)
plt.hist(arpc2ko_motion_df[feature].dropna(),bins=bin_edges,histtype='step',color='red',label='ARPC2KO',density=True)
# plt.title('{}'.format(feature))
plt.title('{} distribution'.format(feature))
plt.xlabel('{}'.format(feature))
plt.ylabel('density')
plt.legend()
plt.savefig('{}/{}_distribution.png'.format(save_path, feature),bbox_inches='tight')
plt.clf()

feature = 'DT'
hist, bin_edges = np.histogram(wt_motion_df[feature].dropna(),bins=15)
plt.hist(wt_motion_df[feature].dropna(),bins=bin_edges,histtype='step',color='blue',label='WT',density=True)
plt.hist(arpc2ko_motion_df[feature].dropna(),bins=bin_edges,histtype='step',color='red',label='ARPC2KO',density=True)
# plt.title('{}'.format(feature))
plt.title('{} distribution'.format(feature))
plt.xlabel('{}'.format(feature))
plt.ylabel('density')
plt.legend()
plt.savefig('{}/{}_distribution.png'.format(save_path, feature),bbox_inches='tight')
plt.clf()