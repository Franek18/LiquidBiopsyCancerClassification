import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    """Cancec Classification dataset.
        As our dataset we use samples stored as a 2D matrices in .txt file
        instead of images.
        The size of every sample is [267, 531].
    """

    def __init__(self, annotations_file, img_dir, group=None, signals_permutation=None, columns_permutation=None, transform=None, target_transform=None):
        '''
            @param annotations_file: file with two columns: Path to the sample in
            directory with samples and Class of this sample.
            @param img_dir: directory in which samples from both classes: cancer
            and non-cancer are stored.
            @param group: parameter used when we want to select from a sample
            rows/signal paths only from a given group.
            @param signals_permutation: permutation of rows in every sample if
            we want to train on permutated rows.
            @param columns_permutation: permutation of columns in every sample if
            we want to train on permutated columns.
            @param transofrm: transoformation of a sample i.e. mean and standard deviation.
            @param target_transform: ransoformation of a label of a sample.
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.group = group
        self.signals_permutation = signals_permutation
        self.columns_permutation = columns_permutation
        self.transform = transform
        self.target_transform = target_transform
        self.train_ids = []
        self.test_ids = []


    def __len__(self):
        '''
            Return number of samples in a dataset.
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
            Return a sample from the given index.
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path, ImageReadMode.RGB)
        #image = Image.open(img_path)#.convert('RGB')
        image = np.loadtxt(img_path)

        if self.group is not None:
            image = image[self.group[0]:self.group[-1] + 1]

        if self.signals_permutation is not None:
            # randomly permutate signal's pathways in a sample
            image = image[self.signals_permutation]

        if self.columns_permutation is not None:
            image = image[:, self.columns_permutation]

        image = transforms.ToTensor()(image).float()

        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image.float())

        if self.target_transform:
            label = self.target_transform(label)

        return (image, label)
