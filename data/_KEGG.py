import os
from collections import namedtuple
from multiprocessing import Pool as ProcPool
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# a tuple to disambiguate between different datasets
_PathsSet = namedtuple('PathsSet', [
    'TRAIN_INDICES_NAME',
    'TEST_INDICES_NAME',
    'ANNOTATIONS_NAME',
    'DATA_DIR_NAME',
    'SINGLE_DATA_FILE_NAME',
    'REDUCED_DATA_NAME_PREFIX',
    'LOGS_DIR_NAME'
])

# KEGG Cancer/nonCancer from 16.11.2021
_KEGG_CNC_16112021_PATHS = _PathsSet(
    'Train_indices_CnC.npy',
    'Test_indices_CnC.npy',
    'Cancer_annotations_mts.csv',
    'KEGG_Pathway_Image',
    None,
    'reduced_kegg',
    'logs'
)

# KEGG Cancer/nonCancer from 22.02.2022
_KEGG_CNC_22022022_PATHS = _PathsSet(
    'Train_indices_CnC_22022022.npy',
    'Test_indices_CnC_22022022.npy',
    'Cancer_annotations_mts_22022022.csv',
    'KEGG_Pathway_Image_22022022',
    None,
    'reduced_kegg_22022022',
    'logs_22022022'
)

# KEGG Cancer/nonCancer for transfer to new cancer types (from Franek)
_KEGG_CNC_22022022_NEW_CANCERS_PATHS = _PathsSet(
    'Train_indices_CnC_22022022_new_cancers.npy',
    'Test_indices_CnC_22022022_new_cancers.npy',
    'Cancer_annotations_mts_22022022_new_cancers.csv',
    'KEGG_Pathway_Image_22022022',
    None,
    'reduced_kegg_22022022',
    'logs_22022022_new_cancers'
)

# KEGG Cancer/nonCancer with different split (from Franek)
_KEGG_CNC_09022022_NEW_HOSPITAL_PATHS = _PathsSet(
    'Train_indices_CnC_09022022_new_hospital.npy',
    'Test_indices_CnC_09022022_new_hospital.npy',
    'Cancer_annotations_mts_09022022_new_hospital.csv',
    'KEGG_Pathway_Image',
    None,
    'reduced_kegg',
    'logs_09022022_new_hospital'
)

# Cancer/nonCancer before KEGG filtration from 15.05.2022
_PRE_KEGG_CNC_15052022_PATHS = _PathsSet(
    'Train_indices_CnC_15052022_pre-kegg.npy',
    'Test_indices_CnC_15052022_pre-kegg.npy',
    'Cancer_annotations_mts_09022022_new_hospital.csv',
    None,
    'Teps_05_2022_all_counts_normalized_prefiltered.tsv',
    'reduced_pre-kegg_15052022',
    'logs_15052022_pre-kegg'
)

# use this variable when referring to the dataset
CURRENT_DATASET_PATHS = _PRE_KEGG_CNC_15052022_PATHS


def print_dataset_paths_info(where):
    print(f'Executing {where} with these dataset paths:')
    print(CURRENT_DATASET_PATHS)


"""
Data loading
"""


def load_matrices_from_txts(paths: np.ndarray, img_dir: str) -> np.ndarray:
    full_paths = [os.path.join(img_dir, path) for path in paths]

    with ProcPool(4) as pool:  # 4 processes seem to work best on dev machine
        return np.array(pool.map(np.loadtxt, full_paths))


def load_matrices_from_single_tsv(path: str) -> np.ndarray:
    df = pd.read_csv(path, delimiter='\t').transpose()
    return df.to_numpy()


def load_labels(annotations_file: str) -> pd.DataFrame:
    return pd.read_csv(annotations_file)


"""
Train/Test split utils for crossvalidation training
"""

_TRAIN_INDICES_PATH = str(Path(__file__).parent / 'indices' / CURRENT_DATASET_PATHS.TRAIN_INDICES_NAME)
_TEST_INDICES_PATH = str(Path(__file__).parent / 'indices' / CURRENT_DATASET_PATHS.TEST_INDICES_NAME)


def _save_indices(train_indices, test_indices):
    np.save(_TRAIN_INDICES_PATH, train_indices)
    np.save(_TEST_INDICES_PATH, test_indices)


def _load_indices():
    train_indices = np.load(_TRAIN_INDICES_PATH)
    test_indices = np.load(_TEST_INDICES_PATH)
    return train_indices, test_indices


def get_split(mats, labels):
    if os.path.exists(_TRAIN_INDICES_PATH) and os.path.exists(_TEST_INDICES_PATH):
        return _load_indices()
    print("Indices not loaded")

    indices = np.arange(len(labels))
    _, _, _, _, train_indices, test_indices = train_test_split(mats, labels, indices, test_size=0.3, stratify=labels)
    _save_indices(train_indices, test_indices)
    return train_indices, test_indices


"""
Data augmentation & preprocessing utils
"""


def permutate_signals(no_signals: int = 181, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    signals = np.arange(no_signals)
    return np.random.permutation(signals)


def apply_signals_permutation(mats: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return mats[:, perm, :]


def normalize_global(mats: np.ndarray) -> np.ndarray:
    return (mats - np.mean(mats)) / np.std(mats)


def calculate_means(mats: np.ndarray) -> np.ndarray:
    return np.mean(mats, 0)


def calculate_stds(mats: np.ndarray) -> np.ndarray:
    return np.std(mats, 0)


def normalize_fw(mats: np.ndarray) -> np.ndarray:
    """
    Normalize by feature using z-score.
    """
    normalized = mats - np.resize(calculate_means(mats), [*mats.shape])
    return normalized / (np.resize(calculate_stds(mats), [*normalized.shape]) + 1e-15)


# ID: Column range, name
GROUPS_INFO = {
    '1': ((0, 87), 'Metabolism'),
    '2': ((87, 102), 'Genetic Information Processing'),
    '3': ((102, 132), 'Environmental Information Processing'),
    '4': ((132, 149), 'Cellular Processes'),
    '5.1': ((149, 170), 'Immune system'),
    '5.2': ((170, 212), 'Organismal Systems - other'),
    '6.1': ((212, 240), 'Human Diseases - cancer'),
    '6.2': ((240, 267), 'Human Diseases - noncancer')
}


def select_columns_group(mats: np.ndarray, group_id: str) -> np.ndarray:
    """
    Select a column group and only keep its content. Important: needs to be done on 2D matrices, before reduction.
    mats: 2D square feature arrays
    group_id: one of IDs mentioned in `GROUPS_INFO`
    return: 2D square feature arrays with some columns dropped
    """
    col_range = range(*GROUPS_INFO[group_id][0])
    return mats[:, col_range, :]


def square_matrices_to_flat(mats: np.ndarray):
    return np.array([mat.flatten() for mat in mats])


def flat_matrices_to_square(mats: np.ndarray) -> np.ndarray:
    """
    Reshape 1D pixels/features arrays to 2D square matrices, filling NA with 0 at the end where necessary.
    Useful for generating reduced datasets for CNN.
    img: 1D features arrays for samples
    return: 2D square features arrays
    """
    proposed_size = int(np.ceil(np.sqrt(mats.shape[1])))
    size_diff = int(proposed_size ** 2 - mats.shape[1])
    if size_diff != 0:
        filling = np.zeros((mats.shape[0], size_diff))
        filled_mats = np.concatenate([mats, filling], 1)
    else:
        filled_mats = mats
    return filled_mats.reshape((filled_mats.shape[0], proposed_size, proposed_size))


def get_variable_features(stds: np.ndarray) -> np.ndarray:
    return np.argwhere(stds > 1.e-12).squeeze()


def reduce_matrices(mats: np.ndarray, stds: np.ndarray):
    """
    Reduce the samples by removing features that are const.
    """
    variable_feats_ind = get_variable_features(stds)
    return mats[:, variable_feats_ind]


def perform_reduction(mats: np.ndarray, out_file_path: str, to_square: bool = False):
    # może trzeba zrobić fix i bez flatten, np.std(mats, (0, 2)), inne to_square, żeby zachować pełne ścieżki
    mats = square_matrices_to_flat(mats)
    stds = calculate_stds(mats)
    mats = reduce_matrices(mats, stds)
    if to_square:
        mats = flat_matrices_to_square(mats)
    np.save(out_file_path, mats)


def load_data_for_reductions() -> np.ndarray:
    labels = load_labels(CURRENT_DATASET_PATHS.ANNOTATIONS_NAME)

    if CURRENT_DATASET_PATHS.DATA_DIR_NAME is not None:
        mats = load_matrices_from_txts(labels.iloc[:, 0], CURRENT_DATASET_PATHS.DATA_DIR_NAME)
    elif CURRENT_DATASET_PATHS.SINGLE_DATA_FILE_NAME is not None:
        mats = load_matrices_from_single_tsv(CURRENT_DATASET_PATHS.SINGLE_DATA_FILE_NAME)
    else:
        raise ValueError('No input specified')

    return mats


def perform_reductions(mats: np.ndarray) -> None:
    reduced_prefix = CURRENT_DATASET_PATHS.REDUCED_DATA_NAME_PREFIX

    # full dataset
    perform_reduction(mats, f'reduced/{reduced_prefix}_1d.npy', False)

    if len(mats.shape) == 2:
        return  # the data from a single file is 2D and the latter methods rely on data being 3D

    perform_reduction(mats, f'reduced/{reduced_prefix}_2d.npy', True)

    # one group
    for group_id in GROUPS_INFO:
        perform_reduction(select_columns_group(mats, group_id), f'reduced/{reduced_prefix}_1d_group_{group_id}.npy', False)
        perform_reduction(select_columns_group(mats, group_id), f'reduced/{reduced_prefix}_2d_group_{group_id}.npy', True)

    # two groups
    done = []
    for group_a in GROUPS_INFO:
        for group_b in GROUPS_INFO:
            if group_a != group_b and (group_b, group_a) not in done:
                selected_data = np.append(
                    select_columns_group(mats, group_a),
                    select_columns_group(mats, group_b),
                    axis=1
                )
                perform_reduction(selected_data, f'reduced/{reduced_prefix}_1d_group_{group_a},{group_b}.npy', False)
                perform_reduction(selected_data, f'reduced/{reduced_prefix}_2d_group_{group_a},{group_b}.npy', True)
                done.append((group_a, group_b))


if __name__ == '__main__':
    print_dataset_paths_info('_KEGG.main')

    mats = load_data_for_reductions()

    if not os.path.exists('reduced'):
        os.mkdir('reduced')

    perform_reductions(mats)
