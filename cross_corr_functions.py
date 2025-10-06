import numpy as np
from scipy.signal import correlate
import pandas as pd

def cross_correlation_analysis(x, y, max_lag=None, n_permutations=1000):
    """
    Compute cross-correlation between x and y with normalization,
    find best lag, and assess significance via permutation test.
    """
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    
    # Normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # Full cross-correlation
    corr = correlate(x, y, mode='full') / n
    lags = np.arange(-n + 1, n)
    
    # Restrict to desired lag window
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[mask]
    lags = lags[mask]
    
    # Find best lag
    best_idx = np.argmax(np.abs(corr))
    best_lag = lags[best_idx]
    best_corr = corr[best_idx]
    
    # Permutation test for significance
    perm_corrs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_corrs.append(np.max(np.abs(correlate(x, y_perm, mode='valid') / n)))
    
    p_value = np.mean(np.array(perm_corrs) >= np.abs(best_corr))
    
    return best_lag, best_corr, p_value, lags, corr


def apply_cross_corr(feature1, feature2, arpc2ko_cells, wt_cells):
    max_lag = 12

    threshold = 0.7

    pos_corr_arpc2ko_cells = []
    neg_corr_arpc2ko_cells = []
    else_corr_arpc2ko_cells = []

    best_lags_arpc2ko = []
    best_corrs_arpc2ko = []
    pvals_arpc2ko = []

    all_lags_arpc2ko = []
    all_corr_arpc2ko = []

    names_arpc2ko = []

    for df in arpc2ko_cells:
        signal1 = df[feature1][:max_lag]
        signal2 = df[feature2][:max_lag]

        names_arpc2ko.append(df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0])))
        
        best_lag, best_corr, pval, lags, corr = cross_correlation_analysis(signal1, signal2)

        best_lags_arpc2ko.append(best_lag)
        best_corrs_arpc2ko.append(best_corr)
        pvals_arpc2ko.append(pval)

        all_lags_arpc2ko.append(lags)
        all_corr_arpc2ko.append(corr)

        if best_corr < 0 and np.abs(best_corr) > threshold:
            neg_corr_arpc2ko_cells.append(df)
        elif np.abs(best_corr) > threshold:
            pos_corr_arpc2ko_cells.append(df)
            else_corr_arpc2ko_cells.append(df)
        else:
            else_corr_arpc2ko_cells.append(df)
        
    pvals_arpc2ko = np.array(pvals_arpc2ko)
    best_corrs_arpc2ko = np.array(best_corrs_arpc2ko)
    best_lags_arpc2ko = np.array(best_lags_arpc2ko)

    pos_corr_wt_cells = []
    neg_corr_wt_cells = []
    else_corr_wt_cells = []

    best_lags_wt = []
    best_corrs_wt = []
    pvals_wt = []

    all_lags_wt = []
    all_corr_wt = []

    names_wt = []

    for df in wt_cells:
        signal1 = df[feature1][:max_lag]
        signal2 = df[feature2][:max_lag]

        names_wt.append(df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0])))
        
        best_lag, best_corr, pval, lags, corr = cross_correlation_analysis(signal1, signal2)

        best_lags_wt.append(best_lag)
        best_corrs_wt.append(best_corr)
        pvals_wt.append(pval)

        all_lags_wt.append(lags)
        all_corr_wt.append(corr)


        if best_corr < 0 and np.abs(best_corr) > threshold:
            neg_corr_wt_cells.append(df)
        elif np.abs(best_corr) > threshold:
            pos_corr_wt_cells.append(df)
            else_corr_wt_cells.append(df)
        else:
            else_corr_wt_cells.append(df)
        
    pvals_wt = np.array(pvals_wt)
    best_corrs_wt = np.array(best_corrs_wt)
    best_lags_wt = np.array(best_lags_wt)

    lags = np.average(all_lags_wt,axis=0)

    all_corr_wt = np.array(all_corr_wt)
    all_corr_arpc2ko = np.array(all_corr_arpc2ko)

    names_wt = np.array(names_wt)
    names_arpc2ko = np.array(names_arpc2ko)

    pooled_poscorr_arpc2ko = pd.concat(pos_corr_arpc2ko_cells, ignore_index=True)
    pooled_negcorr_arpc2ko = pd.concat(neg_corr_arpc2ko_cells, ignore_index=True)
    pooled_else_arpc2ko = pd.concat(else_corr_arpc2ko_cells, ignore_index=True)
    pooled_poscorr_wt = pd.concat(pos_corr_wt_cells, ignore_index=True)
    pooled_negcorr_wt = pd.concat(neg_corr_wt_cells, ignore_index=True)
    pooled_else_wt = pd.concat(else_corr_wt_cells, ignore_index=True)

    predictedstates_negcorr_arpc2ko = []
    for df in neg_corr_arpc2ko_cells:
        predictedstates_negcorr_arpc2ko.append(df['state'].to_numpy())

    predictedstates_else_arpc2ko = []
    for df in else_corr_arpc2ko_cells:
        predictedstates_else_arpc2ko.append(df['state'].to_numpy())

    predictedstates_negcorr_wt = []
    for df in neg_corr_wt_cells:
        predictedstates_negcorr_wt.append(df['state'].to_numpy())

    predictedstates_else_wt = []
    for df in else_corr_wt_cells:
        predictedstates_else_wt.append(df['state'].to_numpy())

    return pooled_poscorr_arpc2ko, pooled_negcorr_arpc2ko, pooled_else_arpc2ko, pooled_poscorr_wt, pooled_negcorr_wt, pooled_else_wt, predictedstates_negcorr_arpc2ko, predictedstates_else_arpc2ko, predictedstates_negcorr_wt, predictedstates_else_wt
