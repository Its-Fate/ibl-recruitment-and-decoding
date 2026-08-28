from iblutil.util import Bunch
import numpy as np
from one.remote.aws import s3_download_file
import os
import pandas as pd
from scipy import sparse
import zipfile

# When running in jupyter set number of threads to 1
os.environ.setdefault('ONE_HTTP_DL_THREADS', '1')

# Initialize ONE API
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international', cache_dir='../data')


# -- Define loading utility functions--
def download_data(event):
    assert event in ['firstMove', 'stimOn', 'feedback'], 'event must be one of "firstMove", "stimOn" or "feedback'

    # Dataset name
    fname = f'data_{event}.zip'
    # Remote location of data
    s3_data_path = f'sample_data/Neuromatch/{fname}'
    # Local location to download data to
    save_path = one.cache_dir.joinpath('Neuromatch', fname)
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # Download file
    file = s3_download_file(s3_data_path, save_path)
    # Unzip content
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(save_path.parent)


def get_data_path(event):

    return one.cache_dir.joinpath('Neuromatch', f'data_{event}')


def load_metadata(event):
    metadata = Bunch()
    data_path = get_data_path(event)
    metadata['clusters'] = pd.read_parquet(data_path.joinpath('clusters.pqt'))
    metadata['trials'] = pd.read_parquet(data_path.joinpath('trials.pqt'))
    metadata['sessions'] = pd.read_parquet(data_path.joinpath('sessions.pqt'))
    metadata['times'] = np.load(data_path.joinpath('t.npy'))
    metadata['nbins'] = metadata['times'].size
    metadata['dt'] = np.round(np.median(np.diff(metadata['times'])), 2)
    metadata['data_path'] = data_path

    return metadata


def load_times(data_path):
    return np.load(data_path.joinpath('t.npy'))


def load_psth(data_path, pid, nbins=150):
    psth = sparse.load_npz(data_path.joinpath(f'{pid}.npz')).toarray()
    psth = psth.reshape(psth.shape[0], -1, nbins)
    return psth


# -- Define processing utility functions --
def split_trials_by_variable(trials, split='contrast'):
    trials = trials.set_index('psth_index')
    if split == 'contrast':
        trials['contrast'] = np.nansum([trials['contrastLeft'], trials['contrastRight']], axis=0) * 100
        grp = trials.groupby('contrast')
    elif split == 'signed contrast':
        trials['signedContrast'] = np.nansum([-1 * trials['contrastLeft'], trials['contrastRight']], axis=0) * 100
        grp = trials.groupby('signedContrast')
    elif split == 'stimulus':
        trials['stimulus'] = 'right'
        trials.loc[trials['contrastRight'].isna(), 'stimulus'] = 'left'
        grp = trials.groupby('stimulus')
    elif split == 'choice':
        grp = trials.groupby('choice')
    elif split == 'block':
        grp = trials.groupby('probabilityLeft')
    else:
        raise NotImplementedError('split must be one of "contrast", "signed contrast", "stimulus", "choice" or "block"')

    return grp.groups


def get_avg_psth_for_insertion(pid, meta, reg=None, uuids=None, split=None):
    df = meta.clusters[meta.clusters['pid'] == pid]
    df = df[['acronym', 'pid', 'uuids', 'cluster_id', 'psth_index']]
    sp = load_psth(meta.data_path, pid, nbins=meta.nbins)

    if reg is not None:
        in_reg = df['acronym'] == reg
        sp = sp[:, in_reg.values, :]
        df = df[in_reg].reset_index(drop=True)

    if uuids is not None:
        in_uuid = df['uuids'].isin(uuids)
        sp = sp[:, in_uuid.values, :]
        df = df[in_uuid].reset_index(drop=True)

    if split is None:
        psth = sp.mean(axis=0) / meta['dt']
    else:
        psth = dict()
        eid = meta.sessions[meta.sessions['pid'] == pid].iloc[0]['eid']
        trials = meta.trials[meta.trials['eid'] == eid].reset_index(drop=True)
        grps = split_trials_by_variable(trials, split=split)

        for key, vals in grps.items():
            psth[key] = sp[vals, :, :].mean(axis=0)

    return psth, df

def get_avg_psth_for_region(reg, meta, split=None):
    clusters = meta.clusters[meta.clusters['acronym'] == reg]
    pids = clusters['pid'].unique()
    all_df = []
    all_psth = []
    for pid in pids:
        psth, df = get_avg_psth_for_insertion(pid, meta, reg=reg, split=split)
        all_df.append(df)
        all_psth.append(psth)

    all_df = pd.concat(all_df).reset_index(drop=True)
    if split is None:
        all_psth = np.concatenate(all_psth)
    else:
        all_psth = {key: np.concatenate([d[key] for d in all_psth if key in d.keys()])
        for key in all_psth[0]}


    return all_psth, all_df


def get_avg_psth_for_clusters(uuids, meta, split=None):
    clusters = meta.clusters[meta.clusters['uuids'].isin(uuids)]
    pids = clusters['pid'].unique()
    all_df = []
    all_psth = []
    for pid in pids:
        psth, df = get_avg_psth_for_insertion(pid, meta, uuids=uuids, split=split)
        all_df.append(df)
        all_psth.append(psth)

    all_df = pd.concat(all_df).reset_index(drop=True)
    if split is None:
        all_psth = np.concatenate(all_psth)
    else:
        all_psth = {key: np.concatenate([d[key] for d in all_psth if key in d.keys()])
        for key in all_psth[0]}

    return all_psth, all_df



def get_psth_for_insertion(pid, meta, reg=None, uuids=None):
    df = meta.clusters[meta.clusters['pid'] == pid]
    df = df[['acronym', 'pid', 'uuids', 'cluster_id', 'psth_index']]
    sp = load_psth(meta.data_path, pid, nbins=meta.nbins)

    if reg is not None:
        in_reg = df['acronym'] == reg
        sp = sp[:, in_reg.values, :]
        df = df[in_reg].reset_index(drop=True)

    if uuids is not None:
        in_uuid = df['uuids'].isin(uuids)
        sp = sp[:, in_uuid.values, :]
        df = df[in_uuid].reset_index(drop=True)


    eid = meta.sessions[meta.sessions['pid'] == pid].iloc[0]['eid']
    trials = meta.trials[meta.trials['eid'] == eid].reset_index(drop=True)
    psth = sp / meta['dt']

    return psth, df, trials


def get_psth_for_region(reg, meta):
    clusters = meta.clusters[meta.clusters['acronym'] == reg]
    pids = clusters['pid'].unique()
    all_clust = []
    all_psth = []
    all_trials = []
    for pid in pids:
        psth, clust, trials = get_psth_for_insertion(pid, meta, reg=reg)
        all_clust.append(clust)
        all_psth.append(psth)
        all_trials.append(trials)

    return all_psth, all_clust, all_trials


def get_psth_for_clusters(uuids, meta):
    clusters = meta.clusters[meta.clusters['uuids'].isin(uuids)]
    pids = clusters['pid'].unique()
    all_clust = []
    all_psth = []
    all_trials = []
    for pid in pids:
        psth, clust, trials = get_psth_for_insertion(pid, meta, uuids=uuids)
        all_clust.append(clust)
        all_psth.append(psth)
        all_trials.append(trials)

    return all_psth, all_clust, all_trials


def create_side_contrast_table(side_summary, contrast_summary):
    side = side_summary.rename(columns={
        'onset': 'side_onset',
        'offset': 'side_offset',
        'total_duration': 'side_duration',
        'peak_score': 'side_peak_score',
        'peak_center': 'side_peak_time'
    })

    contrast = contrast_summary.rename(columns={
        'onset': 'contrast_onset',
        'offset': 'contrast_offset',
        'total_duration': 'contrast_duration',
        'peak_score': 'contrast_peak_score',
        'peak_center': 'contrast_peak_time'
    })

    comparison = side.merge(
        contrast,
        on='region',
        how='outer'
    )

    comparison = comparison[
        [
            'region',
            'side_onset',
            'contrast_onset',
            'side_offset',
            'contrast_offset',
            'side_duration',
            'contrast_duration',
            'side_peak_score',
            'contrast_peak_score',
            'side_peak_time',
            'contrast_peak_time'
        ]
    ]

    comparison = comparison.sort_values('region').reset_index(drop=True)

    return comparison