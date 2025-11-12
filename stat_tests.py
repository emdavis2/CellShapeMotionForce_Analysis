import numpy as np

def paired_preserving_permutation_test(per_cell_state_df, stateA, stateB, n_perm=10000, rng_seed=0):
    """
    per_cell_state_df: dataframe with columns ['cell_id','state','value']
    stateA/stateB: labels of the two states to compare
    returns: observed_diff, p_value, perm_diffs (np.array)
    """
    rng = np.random.default_rng(rng_seed)

    # pivot to wide
    wide = per_cell_state_df.pivot(index='track_name', columns='state_name', values='value')


    # ensure columns exist
    for s in (stateA, stateB):
        if s not in wide.columns:
            wide[s] = np.nan

    paired_idx = wide.index[wide[stateA].notna() & wide[stateB].notna()]
    A_only_idx = wide.index[wide[stateA].notna() & wide[stateB].isna()]
    B_only_idx = wide.index[wide[stateB].notna() & wide[stateA].isna()]

    # observed group statistics (choose median difference; you can use mean)
    groupA_obs = np.concatenate([wide.loc[paired_idx, stateA].values, wide.loc[A_only_idx, stateA].values])
    groupB_obs = np.concatenate([wide.loc[paired_idx, stateB].values, wide.loc[B_only_idx, stateB].values])
    obs_stat = np.median(groupB_obs) - np.median(groupA_obs)

    perm_stats = np.empty(n_perm)
    # Pre-extract arrays
    A_paired = wide.loc[paired_idx, stateA].values
    B_paired = wide.loc[paired_idx, stateB].values
    A_unpaired = wide.loc[A_only_idx, stateA].values
    B_unpaired = wide.loc[B_only_idx, stateB].values

    for i in range(n_perm):
        # For paired cells: randomly swap A/B or keep
        swap = rng.choice([0,1], size=len(A_paired))
        A_p = np.where(swap==0, A_paired, B_paired)
        B_p = np.where(swap==0, B_paired, A_paired)

        # For unpaired: pool and reshuffle labels among unpaired slots
        pool_unpaired = np.concatenate([A_unpaired, B_unpaired])
        rng.shuffle(pool_unpaired)
        A_up = pool_unpaired[:len(A_unpaired)]
        B_up = pool_unpaired[len(A_unpaired):]

        # Combined groups
        groupA = np.concatenate([A_p, A_up]) if len(A_p)>0 or len(A_up)>0 else np.array([])
        groupB = np.concatenate([B_p, B_up]) if len(B_p)>0 or len(B_up)>0 else np.array([])

        # If either group empty (edge case), skip
        if len(groupA)==0 or len(groupB)==0:
            perm_stats[i] = np.nan
        else:
            perm_stats[i] = np.median(groupB) - np.median(groupA)

    perm_stats = perm_stats[~np.isnan(perm_stats)]
    p_val = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
    return obs_stat, p_val, perm_stats

