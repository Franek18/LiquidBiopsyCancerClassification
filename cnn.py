import os
import csv
import torch
import numpy as np
import torchvision.transforms as transforms

from hparams import hparams
from data import CancerDataset
from torchsummary import summary
from annotate import annotateMultiClass
from cnn_training import kFoldTraining
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def summaryModel(model, tensor_shape):
    '''
        Display summary of a model.
        @param model: initialized NN model.
        @param tensor_shape: shape of a sample being a PyTorch tensor.
    '''
    print("Summary of a model")
    summary(model, tensor_shape)

def get_mean_std_labels(dataset):
    '''
        Calculates mean and std for the given dataset.
        It also gathers a vector of labels from a dataset.
        @param dataset: instance of a Class CancerDataset.
        @return mean: mean from all of the dataset.
        @return std: std from all of the dataset.
        @return labels: vector of labels from a given dataset.
    '''
    loader = DataLoader(dataset, batch_size=1)
    mean = 0.
    std = 0.
    labels = []
    for data, label in loader:
        mean += torch.mean(data)
        std += torch.std(data)
        labels.append(label.item())

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std, labels

def save_indices(train_filename, test_filename, train_indices, test_indices):
    '''
        Save indices of samples from train and test sets to the numpy .npy files.
        @param train_filename: path to the .npy file for train set.
        @param test_filename: path to the .npy file for test set.
        @param train_indices: indices of samples from a train set.
        @param test_indices: indices of samples from a test set.
    '''
    np.save(train_filename, train_indices)
    np.save(test_filename, test_indices)

def load_indices(train_filename, test_filename):
    '''
        Load indices of train and test samples from tnumpy .npy files.
        @param train_filename: path to the .npy file for train set.
        @param test_filename: path to the .npy file for test set.
        @return train_indices: indices of samples from a train set.
        @return test_indices: indices of samples from a test set.
    '''
    train_indices = np.load(train_filename)
    test_indices = np.load(test_filename)

    return train_indices, test_indices

def split_multiclass_set(annotation_filename, train_filename, test_filename, other_filename="indices/other_indices.npy"):
    '''
    '''
    train_indices = []
    test_indices = []
    other_indices = []
    with open(annotation_filename, "r") as annotation_file:
        annotation_reader = csv.DictReader(annotation_file, delimiter=',')
        idx = 0
        for row in annotation_reader:
            if idx == 0:
                idx += 1
                continue

            split_id = row["Split"]

            if split_id == "Train":
                train_indices.append(idx - 1)
            elif split_id == "Test":
                test_indices.append(idx - 1)
            else:
                other_indices.append(idx - 1)
            idx += 1

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    other_indices = np.array(other_indices)

    np.save(train_filename, train_indices)
    np.save(test_filename, test_indices)
    np.save(other_filename, other_indices)

    return train_indices, test_indices, other_indices

def train(train_filename, test_filename, result_filename, log_filename, write_mode, lr_list, Dropout_list, wd_list, no_folds = 5):
    '''
        Training of a model.
        @param train_filename:
        @param test_filename:
        @param result_filename:
        @param log_filename:
        @param write_mode:
        @param lr_list:
        @param Dropout_list:
        @param wd_list:
        @param no_folds:
    '''
    # std and mean normalization if we use transfer learning from Imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Type pathway to the directory with your data
    data_dir = ""
    annotation_file = 'annotations/MultiClass_annotations_mts.csv'
    dataset_norm = CancerDataset(annotation_file, data_dir)
    # normalization of data
    mean, std, labels = get_mean_std_labels(dataset_norm)
    normalize = transforms.Normalize(mean,std)

    # Create CancerDataset object with calculated normalization
    dataset = CancerDataset(annotation_file, data_dir, transform=normalize)

    # Load or create indices for train and test sets
    train_indices = None
    test_indices = None

    if not os.path.exists(train_filename):
        # indices = np.arange(len(labels))
        # train_indices, test_indices = train_test_split(indices, test_size=0.3, stratify=labels)
        # save_indices(train_filename, test_filename, train_indices, test_indices)
        train_indices, test_indices, other_indices = split_multiclass_set(annotation_file, train_filename, test_filename)
    else:
        train_indices, test_indices = load_indices(train_filename, test_filename)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    y_train = np.array(labels)[train_indices]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)


    # Get test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Count class wieghts
    weights = compute_class_weight(class_weight = 'balanced', classes = [0, 1, 2, 3, 4, 5, 6], y = labels)
    class_weights = torch.Tensor(weights)

    # Open the file for results
    result_file = open(result_filename, write_mode)
    # Open the file for logs
    log_file = open(log_filename, write_mode)


    # Names of parameters reported in result file
    fieldnames = ["Learning_rate", "Dropout", "weight_decay", "class_weight1",
            "class_weight2", "mixup_alpha"]

    # Create columns for result of each fold
    for i in range(no_folds):
        fieldnames.append("fold_" + str(i) + "_val_auc")
    for i in range(no_folds):
        fieldnames.append("fold_" + str(i) + "_test_auc")

    fieldnames.append("mean_val_auc")
    fieldnames.append("mean_test_auc")

    # Get DictWriter
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)

    # Check the file mode, write headers of columns if it is a write mode
    if write_mode == "w":
        writer.writeheader()


    # Train models in k-fold training
    for name in ["ResNet18"]:
        for lr in lr_list:
            for Dropout in Dropout_list:
                for wd in wd_list:
                    hparams[name]["lr"] = lr
                    hparams[name]["weight_decay"] = wd
                    hparams[name]["Dropout"] = Dropout
                    #model = get_ResNet(Dropout, no_layers=hparams[name]["no_layers"], pooling=False)
                    log_file.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                    log_file.write("Training on:\n")
                    log_file.write("Lr = {}\n".format(lr))
                    log_file.write("Dropout = {}\n".format(Dropout))
                    log_file.write("weight_decay = {}\n".format(wd))
                    log_file.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

                    kFoldTraining(log_file, no_folds, writer, y_train, train_dataset, test_dataloader, test_indices, class_weights, hparams[name], no_classes=7)

    result_file.close()
    log_file.close()

def main():

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('annotations'):
        os.makedirs('annotations')
    if not os.path.exists('indices'):
        os.makedirs('indices')

    if not os.path.exists('annotations/MultiClass_annotations_mts.csv'):
        annotateMultiClass()

    #lr_list = np.logspace(-1, -3, 10)
    # hyperparameters
    lr_list = [0.001]
    Dropout_list = [0.4]
    wd_list = [1e-3]

    train_filename = "indices/train_indices.npy"
    test_filename = "indices/test_indices.npy"
    result_file = "results/Multiclass_lr001_dr04_wd001.csv"
    log_file = "logs/Multiclass_lr001_dr04_wd001.txt"
    write_mode = "w"

    for i in range(3):
        train(train_filename, test_filename, result_file, log_file, write_mode, lr_list, Dropout_list, wd_list)
        write_mode = "a"

if __name__ == "__main__":
    main()
