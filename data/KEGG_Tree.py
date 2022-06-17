from pathlib import Path
from typing import Tuple, Optional

import numpy as np

import data._KEGG as k


def preprocess_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = k.square_matrices_to_flat(x)
    train_i, test_i = k.get_split(X, y)
    return X[train_i], y[train_i], X[test_i], y[test_i]


def load_original_data(col_group: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = k.load_labels(str(Path(__file__).parent / k.CURRENT_DATASET_PATHS.ANNOTATIONS_NAME))
    X = k.load_matrices(labels.iloc[:, 0].to_numpy(), str(Path(__file__).parent / k.CURRENT_DATASET_PATHS.DATA_DIR_NAME))
    if col_group is not None:
        X = k.select_columns_group(X, col_group)
    y = np.array(labels.iloc[:, 1])
    return preprocess_data(X, y)


def load_reduced_data(col_group: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mats_file = f'reduced/{k.CURRENT_DATASET_PATHS.REDUCED_DATA_NAME_PREFIX}_1d{f"_group_{col_group}" if col_group is not None else ""}.npy'
    X = np.load(str(Path(__file__).parent / mats_file))
    labels = k.load_labels(str(Path(__file__).parent / k.CURRENT_DATASET_PATHS.ANNOTATIONS_NAME))
    y = np.array(labels.iloc[:, 1])
    return preprocess_data(X, y)


def extract_most_important_features(x: np.ndarray, feature_importances: np.ndarray, threshold: Optional[float] = None, feat_num: Optional[int] = None) -> np.ndarray:
    if (threshold is None) == (feat_num is None):
        raise ValueError("Must use 1 and only 1 kind of threshold")
    sorted_ind = feature_importances.argsort()
    if threshold is not None:
        imp_features_num = int(len(feature_importances) * threshold)
        imp_features = sorted_ind[-imp_features_num:]
    else:
        imp_features = sorted_ind[-feat_num:]
    return x[:, imp_features]
