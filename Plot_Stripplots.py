import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu

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
   "protrusion_lengths": parse_list,
    "median_protrusion_widths": parse_list,
    "mean_protrusion_widths": parse_list
})

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, skeleton_df, protrusion_df)

numeric_pooled_arpc2ko_df = pooled_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_wt_df.select_dtypes(include=np.number)

# save_path = './figures/stripplots'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

for df in arpc2ko_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)

for df in wt_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_wt_df = pd.concat(wt_cells, ignore_index=True)

# data_bp = pd.concat([pooled_arpc2ko_df, pooled_wt_df]).reset_index()

# for column in numeric_pooled_arpc2ko_df.columns:
#     sns.stripplot(data=data_bp, x='type', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
#     sns.pointplot(data=data_bp, x="type", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
#     data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].dropna()
#     data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].dropna()
#     U1, p = mannwhitneyu(data1, data2)
#     plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,2)))
#     plt.legend()
#     plt.xlabel("Type")
#     plt.ylabel("{}".format(column))
#     plt.xticks(rotation=90)

#     plot_name = '{}_stripplot'.format(column)

#     plt.savefig('{}/{}.png'.format(save_path, plot_name),bbox_inches='tight')
#     plt.clf()

pixel_size = 0.645 #microns
motion_df = {'speed':[], 'DT':[], 'type':[], 'track_name':[]}

wt_median_summary_df = [] 
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

    motion_df['speed'].append(speed)
    motion_df['DT'].append(DT)
    motion_df['type'].append(df['type'].iloc[0])
    motion_df['track_name'].append(df['track_name'].iloc[0])

    medians = df.median(numeric_only = True)        
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    wt_median_summary_df.append(medians)

wt_median_summary_df = pd.concat(wt_median_summary_df,axis=1).T

arpc2ko_median_summary_df = []
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

    motion_df['speed'].append(speed)
    motion_df['DT'].append(DT)
    motion_df['type'].append(df['type'].iloc[0])
    motion_df['track_name'].append(df['track_name'].iloc[0])

    medians = df.median(numeric_only = True)        
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    arpc2ko_median_summary_df.append(medians)

arpc2ko_median_summary_df = pd.concat(arpc2ko_median_summary_df,axis=1).T
motion_df = pd.DataFrame(motion_df)


save_path = './figures/stripplots_median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

column = 'speed'
# sns.stripplot(data=motion_df, x='type', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
# sns.pointplot(data=motion_df, x="type", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
sns.violinplot(data=motion_df, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
sns.swarmplot(data=motion_df, x='type', y=column, color=(243/256,243/256,243/256),legend=False)
data1 = motion_df[motion_df['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
data2 = motion_df[motion_df['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.xlabel("Type")
plt.ylabel("{}".format(column))
plt.legend()
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format(column)
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

column = 'DT'
# sns.stripplot(data=motion_df, x='type', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
# sns.pointplot(data=motion_df, x="type", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
sns.violinplot(data=motion_df, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
sns.swarmplot(data=motion_df, x='type', y=column, color=(243/256,243/256,243/256),legend=False)
data1 = motion_df[motion_df['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
data2 = motion_df[motion_df['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.xlabel("Type")
plt.ylabel("{}".format(column))
plt.legend()
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format(column)
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

# save_path = './figures/stripplots_median'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)

# numeric_arpc2ko_median_df = arpc2ko_median_summary_df.select_dtypes(include=np.number)

# data_bp = pd.concat([arpc2ko_median_summary_df, wt_median_summary_df]).reset_index()
# for column in numeric_pooled_arpc2ko_df.columns:
#     sns.stripplot(data=data_bp, x='type', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
#     sns.pointplot(data=data_bp, x="type", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
#     data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
#     data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
#     U1, p = mannwhitneyu(data1, data2)
#     plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,2)))
#     plt.legend()
#     plt.xlabel("Type")
#     plt.ylabel("{}".format(column))
#     plt.xticks(rotation=90)

#     plot_name = '{}_stripplot'.format(column)

#     plt.savefig('{}/{}.png'.format(save_path, plot_name),bbox_inches='tight')
#     plt.clf()


params = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','turning_angle','step_length']

save_path = './figures/stripplots_median/pdf_vp'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

numeric_arpc2ko_median_df = arpc2ko_median_summary_df.select_dtypes(include=np.number)

data_bp = pd.concat([wt_median_summary_df, arpc2ko_median_summary_df]).reset_index()
for column in params: #numeric_pooled_arpc2ko_df.columns:
    # sns.stripplot(data=data_bp, x='type', y=column, hue='track_name',alpha=0.7,palette='deep',legend=False)
    # sns.pointplot(data=data_bp, x="type", y=column,linestyle="none",marker="_",markersize=50,capsize=.2,markeredgewidth=3,color=".5",errorbar='sd')
    sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
    sns.swarmplot(data=data_bp, x='type', y=column, color=(243/256,243/256,243/256),legend=False)
    data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
    data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
    U1, p = mannwhitneyu(data1, data2)
    plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
    plt.legend()
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.xticks(rotation=90)

    plot_name = '{}_stripplot'.format(column)

    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()