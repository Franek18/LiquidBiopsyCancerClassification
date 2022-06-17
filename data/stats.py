from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import data._KEGG as k


def plot_stds_heatmap():
    labels = k.load_labels(str(Path(__file__).parent / 'Cancer_annotations_mts.csv'))
    X = k.load_matrices(labels.iloc[:, 0], str(Path(__file__).parent / 'KEGG_Pathway_Image'))
    stds = k.calculate_stds(X)
    sns.heatmap(stds)
    plt.show()


if __name__ == "__main__":
    plot_stds_heatmap()
