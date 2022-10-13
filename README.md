### Liquid Biopsy Cancer Classification

This is the official code repository to the paper [Platelet RNA sequencing data through the lens of machine learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174793).

#### Used libraries
```
Python==3.8.10
torch==1.7.1
torchvision==0.8.2
scikit-learn==0.24.1
```

#### How to start
First create Python3 virtual environment:
```
python3 -m venv venv
```
Then install all required Python packages:
```
pip install -r requirements.txt
```

The main file is `cnn.py`. In this file we must set such variables like:
- **lr_list** - list with at least one value of learning rate for our model
- **Dropout_list** - list with at least one value of Dropout of final layer for our model
- **wd_list** - list with at least one value of weight decay for optimizer of our model
- **train_filename** - name of numpy file with indices to the train samples from whole dataset. If doesn't exist it will be created a new file, with new train/test split.
- **test_filename** - name of numpy file with indices to the test samples from whole dataset. If doesn't exist it will be created a new file, with new train/test split.
- **result_file** - name of csv file for saving results like val and test auc for each fold.
- **log_file** - name of txt file for logs from training of a model.

In the same file, in function **train** we must set variable **data_dir** and **annotation_file** to the locations of dataset and file with annotations.
It should be equal:
```commandline
'annotations/Cancer_annotations_mts.csv'
```
which is set by function **annotateCancer**.
