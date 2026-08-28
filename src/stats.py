import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def statistical_analysis(results_df, label, min_session=2, alpha=0.05):
    '''
    Groups by region and performs a statistical analysis of decodind results.

    alpha: float
        FDR significance threshold

    returns --> stats_df: pd.DataFrame
        (region X time)
    '''

    if label == 'contrast':
        chance = 0.2
    elif label == 'side':
        chance = 0.5
    else:
        raise ValueError('Label should be either "contrast" or "side"')

    # Region-level aggregation (here we go again) 
    # but this time with columns specific to statistical tests
    rl_agg = (results_df.groupby(['region', 'center']).agg(
        n_sessions=('score', 'count'),
        mean_score=('score', 'mean'),
        scores=('score', list)
    )).reset_index()

    # Keep only the region x time points with enough sessions
    stats_df = rl_agg[rl_agg['n_sessions'] >= min_session].copy()

    # Perform one-sample t-test against chance
    t_stats = []
    p_values = []
    for scores in stats_df['scores']:
        scores = np.asarray(scores)

        t_stat, p_value = ttest_1samp(scores, popmean=chance)

        t_stats.append(t_stat)
        p_values.append(p_value)

    stats_df['t_stat'] = t_stats
    stats_df['p_value'] = p_values

    # FDR (False Discovery Rate) correction across all region × time tests
    reject, p_fdr, _, _ = multipletests(stats_df['p_value'], alpha=0.05, method='fdr_bh')

    stats_df['p_fdr'] = p_fdr
    stats_df['significant'] = reject

    # Add above_chance and sig_above_chance columns
    stats_df['above_chance'] = stats_df['mean_score'] > chance
    stats_df['sig_above_chance'] = stats_df['above_chance'] & stats_df['significant']

    # We don't need the scores list column
    stats_df = stats_df.drop(columns='scores')

    return stats_df


def extract_significant_periods(stats_df, win=0.1, min_consecutive=2):
    '''Return one row per significant period'''

    periods = []

    for region, group in stats_df.groupby('region'):
        group = group.sort_values('center').reset_index(drop=True)

        centers = group['center'].to_numpy()
        significant = group['sig_above_chance'].to_numpy()

        start_idx = None

        for i, is_sig in enumerate(significant):

            # Start a new significant period
            if is_sig and start_idx is None:
                start_idx = i

            # End a significant period
            elif not is_sig and start_idx is not None:
                n_bins = i - start_idx

                if n_bins >= min_consecutive:
                    start_time = centers[start_idx] - win / 2
                    end_time = centers[i - 1] + win / 2

                    periods.append({
                        'region': region,
                        'start': start_time,
                        'end': end_time,
                        'n_bins': n_bins,
                        'duration': end_time - start_time
                    })

                # Reset index
                start_idx = None

        # Handling the case a period is significant up until the end
        if start_idx is not None:
            n_bins = len(significant) - start_idx

            if n_bins >= min_consecutive:
                start_time = centers[start_idx] - win / 2
                end_time = centers[-1] + win / 2

                periods.append({
                    'region': region,
                    'start': start_time,
                    'end': end_time,
                    'n_bins': n_bins,
                    'duration': end_time - start_time
                })

    return pd.DataFrame(periods)


def summarize_decoding(stats_df, sig_periods, win=0.1):
    summaries = []

    for region, group in stats_df.groupby('region'):

        # Significant periods for this region
        periods = sig_periods[sig_periods['region'] == region]

        if periods.empty:
            continue

        # Earliest onset
        onset = periods['start'].min()

        # Latest offset
        offset = periods['end'].max()

        # Total duration of significant periods
        total_duration = periods['duration'].sum()

        # Peak decoding
        peak_idx = group['mean_score'].idxmax()
        peak_score = group.loc[peak_idx, 'mean_score']
        peak_time = group.loc[peak_idx, 'center']

        summaries.append({
            'region': region, 
            'onset': onset,
            'offset': offset,
            'total_duration': total_duration,
            'peak_score': peak_score,
            'peak_center': peak_time, 
            'peak_time': f'{round((peak_time - win / 2), 4)} - {round((peak_time + win / 2), 4)}'
        })

    return pd.DataFrame(summaries)