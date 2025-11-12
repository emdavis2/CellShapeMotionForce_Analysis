import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import umap
from scipy.stats import norm

from get_data_functions import *
from assemble_data_functions import *

sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14,10)})

save_path = './figures/UMAP_WTOnly'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

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


pooled_df = pd.concat(wt_cells, ignore_index=True)

params = ["step_length",
        "eccentricity",
        "area",
        "solidity"]

cell_data = pooled_df[params].values
scaled_cell_data = StandardScaler().fit_transform(cell_data)

reducer = umap.UMAP(random_state=66)

embedding = reducer.fit_transform(scaled_cell_data)
embedding.shape

pooled_df['umap_x'] = embedding[:, 0]
pooled_df['umap_y'] = embedding[:, 1]

pooled_wt_df = pooled_df[pooled_df['type'] == 'WT']



param = 'avg_trac_mag'
plt.scatter(pooled_df['umap_x'], pooled_df['umap_y'], c=pooled_df[param], cmap="plasma")
plt.title('UMAP projection of the cell dataset - colored by {}'.format(param), fontsize=24)
cbar = plt.colorbar()
cbar.set_label('{}'.format(param))
plt.savefig('{}/{}_umap.png'.format(save_path, param),bbox_inches='tight')
plt.clf()


for param in params:
    plt.scatter(pooled_df['umap_x'], pooled_df['umap_y'], c=pooled_df[param], cmap="plasma")
    plt.title('UMAP projection of the cell dataset - colored by {}'.format(param), fontsize=24)
    cbar = plt.colorbar()
    cbar.set_label('{}'.format(param))
    plt.savefig('{}/{}_umap.png'.format(save_path, param),bbox_inches='tight')
    plt.clf()