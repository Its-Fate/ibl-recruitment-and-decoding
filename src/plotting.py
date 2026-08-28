import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions


def plot_distribution_of_decoding_score(results_df, label):
    if label == 'contrast':
        chance = 0.2
        chance_label = 'chance (0.20)'
    elif label == 'side':
        chance = 0.5
        chance_label = 'chance (0.50)'
    else:
        raise ValueError('Label should be either "contrast" or "side"')

    plt.figure(figsize=(8,4))
    sns.histplot(results_df['score'], bins=50, kde=True)
    plt.axvline(chance, color='red', linestyle='--', label=chance_label)
    plt.xlabel('Balanced accuracy')
    plt.ylabel('Count')
    plt.title('Distribution of decoding scores across all region-session-time')
    plt.legend()
    plt.show()


def plot_region_score_per_session(results_df, regions_to_plot, label):
    if label == 'contrast':
        chance = 0.2
        chance_label = 'chance (0.20)'
    elif label == 'side':
        chance = 0.5
        chance_label = 'chance (0.50)'
    else:
        raise ValueError('Label should be either "contrast" or "side"')

    for reg in regions_to_plot:
        df_reg = results_df[
            results_df['region'] == reg
        ]

        plt.figure(figsize=(10,4))

        for eid, session in df_reg.groupby('eid'):
            session = session.sort_values('center')

            smoothed_score = gaussian_filter1d(
                session['score'].values,
                sigma=1.5
            )

            plt.plot(
                session['center'],
                smoothed_score,
                alpha=0.6
            )

        # Add this inside your loop right after plotting individual session lines:
        mean_score = df_reg.groupby('center')['score'].mean()
        plt.plot(
            mean_score.index, 
            gaussian_filter1d(mean_score.values, sigma=1.5), 
            color='black', 
            linewidth=2.5, 
            label='Region Mean'
        )

        plt.axhline(
            chance,
            color='red',
            linestyle='--',
            label=chance_label
        )

        plt.title(f'{label.capitalize()} decoding over time – {reg} (each line = one session)')
        plt.xlabel('Time from stimOn (s)')
        plt.ylabel('Balanced accuracy')
        plt.ylim(0, 1)
        plt.legend([], [], frameon=False)
        plt.show()


def plot_region_decoding(rl_agg, title, chance, min_sessions=5, selected_regions=None, num_top_region=10, figsize=(12,6), sigma=1):
    
    # Filter by session count
    rl_agg_filtered = rl_agg[rl_agg['count'] >= min_sessions].copy()
    
    if rl_agg_filtered.empty:
        print("No regions meet the minimum session count.")
        return
    
    # If no specific regions, pick the top 10 by average performance
    if selected_regions is None:
        region_avg = rl_agg_filtered.groupby('region')['mean'].mean()
        selected_regions = region_avg.sort_values(
            ascending=False
        ).head(num_top_region).index.tolist()
    
    plt.figure(figsize=figsize)

    for reg in selected_regions:
        df_reg = rl_agg_filtered[
            rl_agg_filtered['region'] == reg
        ].sort_values('center')

        if df_reg.empty:
            continue

        # Smooth mean curve for visualization
        smoothed_mean = gaussian_filter1d(
            df_reg['mean'].values,
            sigma=sigma
        )

        plt.plot(
            df_reg['center'],
            smoothed_mean,
            label=reg,
            linewidth=2
        )

        # Keep original CI
        plt.fill_between(
            df_reg['center'],
            df_reg['ci_lower'],
            df_reg['ci_upper'],
            alpha=0.1
        )
    
    plt.axhline(
        chance,
        color='black',
        linestyle='--',
        label=f'chance ({chance:.2f})'
    )

    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Balanced accuracy')
    plt.title(f'{title} Decoding - Region Averages (region average ± 95% CI)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_group_decoding(reag_grouped, chance, title, figsize=(12,6), sigma=0.9):

    plt.figure(figsize=figsize)

    for group in reag_grouped['group'].unique():
        df_g = reag_grouped[
            reag_grouped['group'] == group
        ].sort_values('center')

        smoothed_mean = gaussian_filter1d(
            df_g['mean'].values,
            sigma=sigma
        )

        plt.plot(
            df_g['center'],
            smoothed_mean,
            label=group,
            linewidth=2.5
        )

        # Keep original CI
        plt.fill_between(
            df_g['center'],
            df_g['ci_lower'],
            df_g['ci_upper'],
            alpha=0.1
        )

    plt.axhline(
        chance,
        color='black',
        linestyle='--',
        label=f'chance ({chance:.2f})'
    )

    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'{title} Decoding - Anatomical Group Averages (region average ± 95% CI)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decoding_heatmap(rl_agg, title, chance, index_col, min_sessions=5, sort_by='peak', figsize=(14, 10), cmap='coolwarm'):

    # Filter by session count
    rl_agg_filtered = rl_agg[rl_agg['count'] >= min_sessions].copy()
    
    if rl_agg_filtered.empty:
        print("No regions meet the minimum session count.")
        return
    
    # Pivot: rows = regions, columns = time, values = mean decoding score
    pivot = rl_agg_filtered.pivot(index=index_col, columns='center', values='mean')
    
    # Sort rows
    if sort_by == 'peak':
        order = pivot.max(axis=1).sort_values(ascending=False).index
    else:  # sort_by == 'mean'
        order = pivot.mean(axis=1).sort_values(ascending=False).index
    
    pivot = pivot.loc[order]
    
    # Clip values at chance for visual clarity (keeps the colour map focused)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=chance,
        vmax=None,
        cbar_kws={'label': 'Balanced accuracy'},
        linewidths=0.5,
        linecolor='lightgray',
        yticklabels=True
    )
    
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel(index_col.capitalize())
    ax.set_title(f'{title} Decoding Heatmap')

    # Format x‑axis labels and rotate x‑axis labels for readability
    time_values = pivot.columns.values
    ax.set_xticks(np.arange(len(time_values)))
    ax.set_xticklabels([f'{t:.3f}' for t in time_values], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return pivot


def plot_significant_periods(sig_periods, title):
    plot_df = sig_periods.sort_values(['region', 'start']).copy()

    regions = plot_df['region'].unique()

    plt.figure(figsize=(12, max(6, len(regions) * 0.22)))

    for y, region in enumerate(regions):
        region_periods = plot_df[plot_df['region'] == region]

        for _, row in region_periods.iterrows():
            plt.plot(
                [row['start'], row['end']],
                [y, y],
                linewidth=6,
                solid_capstyle='butt'
            )

    plt.axvline(0, linestyle='--', linewidth=1)
    plt.yticks(range(len(regions)), regions)
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Region')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -- Plot individual summaries --
def plot_decoding_onset(summary_df, title=None):
    df = summary_df.sort_values('onset')

    fig, ax = plt.subplots(figsize=(7, 13))

    plt.barh(df['region'], df['onset'])

    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Decoding onset (s)')
    plt.ylabel('Region')
    plt.title(f'{title} decoding onset by region')

    plt.tight_layout()
    plt.show()


def plot_peak_decoding(summary_df, title=None):
    df = summary_df.sort_values('peak_score')

    fig, ax = plt.subplots(figsize=(7, 13))

    scatter = ax.scatter(
        df['peak_center'],
        df['region'],
        s=50,
        c=df['peak_score'],
        cmap='viridis',
        alpha=0.85
    )

    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Peak decoding score')

    ax.set_xlabel('Peak decoding latency (s)')
    ax.set_ylabel('Region')

    if title is None:
        title = 'Peak decoding and latency'

    ax.set_title(title)
    ax.grid(
        axis='y',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()


def plot_decoding_duration(summary_df, title=None):
    df = summary_df.sort_values('total_duration')

    fig, ax = plt.subplots(figsize=(7, 13))

    plt.barh(
        df['region'],
        df['total_duration']
    )

    plt.xlabel('Total significant decoding duration (s)')
    plt.ylabel('Region')
    plt.title(f'{title} duration of significant decoding')

    plt.tight_layout()
    plt.show()


# -- Plot comparison summaries --
def plot_onset_comparison(side_summary, contrast_summary, max_regions=None):

    side = side_summary[['region', 'onset']].copy()
    contrast = contrast_summary[['region', 'onset']].copy()

    df = side.merge(
        contrast,
        on='region',
        suffixes=('_side', '_contrast')
    )

    # Sort by average onset
    df['mean_onset'] = df[['onset_side', 'onset_contrast']].mean(axis=1)
    df = df.sort_values('mean_onset')

    if max_regions is not None:
        df = df.tail(max_regions)

    fig, ax = plt.subplots(figsize=(7, 13))

    y = np.arange(len(df))

    # Connecting lines
    for i, row in df.reset_index(drop=True).iterrows():
        ax.plot(
            [row['onset_side'], row['onset_contrast']],
            [i, i],
            color='gray',
            linewidth=1.2,
            alpha=0.6
        )

    ax.scatter(
        df['onset_side'],
        y,
        s=35,
        label='Side'
    )

    ax.scatter(
        df['onset_contrast'],
        y,
        s=35,
        marker='s',
        label='Contrast'
    )

    ax.axvline(0, linestyle='--', linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df['region'])

    ax.set_xlabel('Decoding onset (s)')
    ax.set_ylabel('Region')
    ax.set_title('Decoding onset: side vs contrast')

    ax.legend()
    ax.grid(
        axis='y',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()


def plot_peak_comparison(side_summary, contrast_summary, max_regions=None):

    side = side_summary[
        ['region', 'peak_center', 'peak_score']
    ].copy()

    contrast = contrast_summary[
        ['region', 'peak_center', 'peak_score']
    ].copy()

    df = side.merge(
        contrast,
        on='region',
        suffixes=('_side', '_contrast')
    )

    # Chance-normalized peak decoding
    # Side: 2 classes -> chance = 0.50
    # Contrast: 5 classes -> chance = 0.20
    df['norm_peak_side'] = (
        (df['peak_score_side'] - 0.50) / (1 - 0.50)
    )

    df['norm_peak_contrast'] = (
        (df['peak_score_contrast'] - 0.20) / (1 - 0.20)
    )

    # Sort by average peak latency
    df['mean_peak'] = (
        df['peak_center_side'] +
        df['peak_center_contrast']
    ) / 2

    if max_regions is not None:
        df = df.sort_values('mean_peak').tail(max_regions)
        df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 14))

    y = np.arange(len(df))

    # Connecting lines
    for i, row in df.iterrows():
        ax.plot(
            [row['peak_center_side'], row['peak_center_contrast']],
            [i, i],
            color='gray',
            linewidth=1.5,
            alpha=0.6,
            zorder=1
        )

    # Side
    side_scatter = ax.scatter(
        df['peak_center_side'],
        y,
        s=55,
        c=df['norm_peak_side'],
        cmap='Blues',
        marker='o',
        edgecolor='blue',
        linewidth=0.5,
        alpha=0.6,
        label='Side',
        zorder=3
    )

    # Contrast
    contrast_scatter = ax.scatter(
        df['peak_center_contrast'],
        y,
        s=55,
        c=df['norm_peak_contrast'],
        cmap='Oranges',
        marker='s',
        edgecolor='orange',
        linewidth=0.5,
        alpha=0.8,
        label='Contrast',
        zorder=2
    )

    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=1
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df['region'])

    ax.set_xlabel('Peak decoding latency (s)')
    ax.set_ylabel('Region')
    ax.set_title('Peak decoding latency: side vs contrast')

    ax.legend()

    ax.grid(
        axis='y',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4
    )

    # Separate colorbars
    side_cbar = plt.colorbar(
        side_scatter,
        ax=ax,
        fraction=0.05,
        pad=0.1
    )
    side_cbar.set_label('Normalized side decoding')

    contrast_cbar = plt.colorbar(
        contrast_scatter,
        ax=ax,
        fraction=0.05,
        pad=0.05
    )
    contrast_cbar.set_label('Normalized contrast decoding')

    plt.tight_layout()
    plt.show()


def plot_duration_comparison(side_summary, contrast_summary, max_regions=None):
    side = side_summary[
        ['region', 'total_duration']
    ].copy()

    contrast = contrast_summary[
        ['region', 'total_duration']
    ].copy()

    df = side.merge(
        contrast,
        on='region',
        suffixes=('_side', '_contrast')
    )

    # Sort by mean duration
    df['mean_duration'] = (
        df['total_duration_side'] +
        df['total_duration_contrast']
    ) / 2

    if max_regions is not None:
        df = df.sort_values('mean_duration').tail(max_regions)
        df = df.reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(7, max(6, 14))
    )

    y = np.arange(len(df))

    # Connecting lines
    for i, row in df.iterrows():
        ax.plot(
            [row['total_duration_side'],
             row['total_duration_contrast']],
            [i, i],
            color='gray',
            linewidth=1.5,
            alpha=0.5,
            zorder=1
        )

    # Side
    ax.scatter(
        df['total_duration_side'],
        y,
        s=55,
        marker='o',
        label='Side',
        zorder=3
    )

    # Contrast
    ax.scatter(
        df['total_duration_contrast'],
        y,
        s=55,
        marker='s',
        label='Contrast',
        zorder=2
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df['region'])

    ax.set_xlabel(
        'Total duration of significant decoding (s)'
    )
    ax.set_ylabel('Region')

    ax.set_title(
        'Significant decoding duration: Side vs Contrast'
    )

    ax.legend()

    ax.grid(
        axis='y',
        linestyle=':',
        linewidth=0.8,
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()


def plot_swanson_vector(acronyms, mode='one'):
    br = BrainRegions()

    swanson_indices = np.unique(br.mappings['Swanson'])
    swanson_ac = np.sort(br.acronym[swanson_indices])

    # Check for missing acronyms
    for ac in acronyms:
        if ac not in swanson_ac:
            print(f'{ac} not in Swanson mapping')

    if mode == 'one':
        # Plot the grouped regions
        for ac in acronyms:
            values = np.array([i for i in range(len(ac))])
            
            plot_swanson_vector(
                ac, values, annotate=True, cmap='rainbow',
                annotate_list=ac, empty_color='silver'
            )
    elif mode == 'sep':
        # Plot the seperate regions
        for ac in acronyms:
            ac = [ac]

            plot_swanson_vector(
                ac, np.array([1]), annotate=True, cmap='Pastel1',
                annotate_list=ac, empty_color='silver'
            )
    else:
        raise ValueError('Mode must be either "one" or "sep"')

    plt.show()