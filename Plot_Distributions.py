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
data_dir = '/Users/emae/Desktop/Analysis_Code/data'
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

save_path = '/Users/emae/Desktop/Analysis_Code/Python_Scripts/figures/histograms'

for column in numeric_pooled_arpc2ko_df.columns:
    plot_dist(column, numeric_pooled_arpc2ko_df, numeric_pooled_wt_df, 'ARPC2KO', 'WT', save_path, column + ' distribution')