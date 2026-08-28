import numpy as np
import pandas as pd
from data_utils import get_psth_for_insertion


# Define sliding window for decoding
def make_window_X(psth, times, center, win=0.1):
    mask = (
        (times >= center - win / 2) &
        (times < center + win / 2)
    )

    # trials x neurons
    return psth[:, :, mask].mean(axis=2)

# Preprocess the regions for decoding
def preprocess_region(reg, meta, min_neurons=10, min_trials_per_class=10):

    # Get all PIDs in this region (and exclude invalid)
    region_clusters = meta.clusters[(meta.clusters['acronym'] == reg)].copy()

    region_data = []

    # Get the EID associated with each PID
    pid_eid = meta.sessions[['pid', 'eid']].drop_duplicates()

    # Add EID to each cluster
    region_clusters = region_clusters.merge(pid_eid, on='pid', how='left')
    
    # This will loop through clusters grouped by session
    for eid, eid_clusters in region_clusters.groupby('eid'):
        # Group PIDs by session
        pids = eid_clusters['pid'].unique()

        psth_list = []
        clusters_list = []
        trials = None

        # Collect data from all PIDs in this (region x session)
        for pid in pids:

            psth, clusters, pid_trials = get_psth_for_insertion(pid, meta, reg=reg)

            # Check trial alignment
            if psth.shape[0] != len(pid_trials):
                print(f"{reg}, {eid}, {pid}: trial mismatch, skipping PID")
                continue

            psth_list.append(psth)
            clusters_list.append(clusters)

            # We have same trials for all PIDs in one session
            if trials is None:
                trials = pid_trials

        if len(psth_list) == 0:
            continue

        # Combine neurons across PIDs
        psth = np.concatenate(psth_list, axis=1) # TODO: why the different concatination methods?
        clusters = pd.concat(clusters_list, ignore_index=True)

        # Minimum neuron requiremnet
        if psth.shape[1] < min_neurons:
            continue

        # Side labels (Right = 1 and Left = -1)
        y_side = np.where(trials['contrastRight'].notna(), 1, -1)

        # Contrast labels
        contrast_values = trials['contrastLeft'].fillna(trials['contrastRight']).values
          # Map the 5 contrast levels to categorical classes
        contrast_map = {
            0.0: 0,
            0.0625: 1,
            0.125: 2,
            0.25: 3,
            1.0: 4
        }
        y_contrast = np.array([contrast_map[con] for con in contrast_values])

        # Check whether the labels and PSTH have matching trials
        if len(y_contrast) != len(psth):
            print(f"{reg}, {eid}: label/PSTH mismatch, skipping")

        # Valid trials for both variables
        valid = (
            np.isfinite(y_contrast) & #TODO: what does isfinite do?
            np.isfinite(y_side)
        )

        psth_clean = psth[valid]
        trials_clean = trials.loc[valid].reset_index(drop=True)

        y_side = y_side[valid]
        y_contrast = y_contrast[valid]

        # Check contrast classes
        contrast_counts = pd.Series(y_contrast).value_counts()

        if contrast_counts.min() < min_trials_per_class:
            continue

        # Check side classes
        side_counts = pd.Series(y_side).value_counts()

        if side_counts.min() < min_trials_per_class:
            continue

        region_data.append({
            'region': reg,
            'eid': eid,
            'pid': list(pids),
            'psth': psth_clean,
            'clusters': clusters,
            'trials': trials_clean,
            'side': y_side,
            'contrast': y_contrast
        })

    return region_data

# Region-level Aggregation
def region_level_aggregation(results_df):

    # Group by region and time center, compute mean and SEM
    rl_agg = results_df.groupby(['region', 'center'])['score'].agg(['mean', 'sem', 'count']).reset_index()

    # Compute 95% CI
    rl_agg['ci_lower'] = rl_agg['mean'] - 1.96 * rl_agg['sem']
    rl_agg['ci_upper'] = rl_agg['mean'] + 1.96 * rl_agg['sem']

    return rl_agg

# Group the regions anatomically
def anatomical_grouping(rl_agg, anatomical_regions):

    group_map = {}
    for group, acronyms in anatomical_regions.items():
        for acr in acronyms:
            group_map[acr] = group

    # Add a 'group' column
    rl_agg['group'] = rl_agg['region'].map(group_map)

    # Drop the regions not in mapping
    agg_grouped = rl_agg[rl_agg['group'].notna()].copy()

    # Aggregation by group --> (group, center)
    reagg_grouped = agg_grouped.groupby(['group', 'center']).agg(
        mean=('mean', 'mean'),
        sem=('sem', 'mean'),
        count=('count', 'sum')).reset_index()

    # Compute CI
    reagg_grouped['ci_lower'] = reagg_grouped['mean'] - 1.96 * reagg_grouped['sem']
    reagg_grouped['ci_upper'] = reagg_grouped['mean'] + 1.96 * reagg_grouped['sem']

    return reagg_grouped