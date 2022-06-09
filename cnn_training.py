import copy
import torch
import numpy as np
import torchvision.models as models

from torch import nn
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_ResNet(dropout, no_layers=18, pretrained=False):
    '''
        This function returns a ResNet model with given numbers of layers.
        It adjusts a model for binary classification problem, adds a Dropout layer
        before the output and modifies input layer because our data
        have only one color channel, not 3 as in typical images.
        @param dropout: value of probability p of an element
        to be zeroed in Dropout layer.
        @param no_layers: number of layers in ResNet. It allows 18, 34 and 50
        layers wariants of ResNet architecture.
        @param pretrained: a bool value if we want to use an Imagenet pretrained weights.
        @return resnet: a ResNet class object.
    '''
    global device
    resnet = None
    # Choose no of layers in ResNet
    if no_layers == 34:
        resnet = models.resnet34(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )
    elif no_layers == 50:
        resnet = models.resnet50()
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )
    else:
        resnet = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )

    new_in_channels = 1

    layer = resnet.conv1

    # Creating new Conv2d layer
    new_layer = nn.Conv2d(in_channels=new_in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias)

    # Copying the weights from the old to the new layer
    new_layer.weight = nn.Parameter(layer.weight[:, :new_in_channels, :, :].clone())
    #print(new_layer.weight[1,0,1,1])
    new_layer.weight = new_layer.weight
    resnet.conv1 = new_layer

    resnet = resnet.to(device)
    return resnet

def calc_confusion_matrix(conf_mtx, preds, labels):
    '''
        This function updates given confusion matrix. We do not use inbuilt function
        because we want to print this matrix in log files in our manner.
        @param conf_mtx: confusion matrix, it is updated regularly after each iteration.
        @param preds: batch of predictions from the model.
        @param labels: batch of labels for preds.
    '''
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            conf_mtx["TP"] += 1

        elif pred == 1 and label == 0:
            conf_mtx["FP"] += 1

        elif pred == 0 and label == 0:
            conf_mtx["TN"] += 1

        else:
            conf_mtx["FN"] += 1

def mixup_data(x, y, alpha):
    '''
        Returns mixed inputs and targets
        @param x: data for mixup.
        @param y: labels for mixup.
        @param alpha: mixup parameter uses for calucalting lambda_p
        @return mixed_x, y_a, y_b, lambda_p: new data obtained from mixup augmentation.
    '''

    lambda_p = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lambda_p * x + (1 - lambda_p) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lambda_p

def mixup_criterion(criterion, pred, y_a, y_b, lambda_p):
    return (lambda_p * criterion(pred, y_a) + (1 - lambda_p) * criterion(pred, y_b))

def train_model(log_file, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, mixup, alpha):
    '''
        Function responsible for conducting the training process of a model.
        @param log_file: file for logs of training like train, val losses, metrics
        and confusion matrix for every epoch.
        @param model: trained model.
        @param dataloaders: Pytorch dataloaders of train and val sets.
        @param dataset_size: sizes of each (train, val) sets.
        @param criterion: loss function.
        @param scheduler: scheduler of a learning rate.
        @param num_epochs: number of epochs for training.
        @param mixup: flag if we want to use mixup augmentation.
        @param alpha: mixup parameter value.
        @return model: trained model.
        @return best_auc: best roc auc score on validation dataset.
        @return best_epoch: epoch in which the best validation auc was achieved.
    '''

    best_model = copy.deepcopy(model.state_dict())
    # validation balanced accuracy
    best_auc = 0.0

    best_epoch = 0

    global device

    for epoch in range(num_epochs):
        log_file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_file.write('--------------------\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            all_outputs = []
            all_preds = []
            all_labels = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # confusion matrix
            conf_mtx = {
                "TP" : 0,
                "FP" : 0,
                "TN" : 0,
                "FN" : 0
            }

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if phase == 'train' and mixup:
                    inputs, labels_a, labels_b, lambda_p = mixup_data(inputs, labels, alpha)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    max_outputs = outputs[:, 1]
                    if phase == 'train' and mixup:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lambda_p)
                    else:
                        loss = criterion(outputs, labels)
                        calc_confusion_matrix(conf_mtx, preds, labels)

                    # backward + optimize only if in training phase
                    all_outputs.extend(max_outputs.detach().cpu().numpy())
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            all_outputs = np.array(all_outputs, dtype=float)
            all_preds = np.array(all_preds, dtype=int)
            all_labels = np.array(all_labels, dtype=int)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train' and mixup == True:
                log_file.write('{} Loss: {:.4f}\n'.format(phase, epoch_loss))

            if phase == 'val' or (phase == 'train' and mixup == False):
                try:
                    curr_recall = conf_mtx["TP"] / (conf_mtx["TP"] + conf_mtx["FN"])
                except ZeroDivisionError:
                    curr_recall = 0

                try:
                    curr_specificity = conf_mtx["TN"] / (conf_mtx["TN"] + conf_mtx["FP"])
                except ZeroDivisionError:
                    curr_specificity = 0

                bal_acc = (curr_recall + curr_specificity) / 2

                auc = roc_auc_score(all_labels, all_outputs, average='weighted')

                log_file.write('Confusion matrix: {}\n'.format(conf_mtx))
                log_file.write('{} Loss: {:.4f} Recall: {:.4f} Specificity: {:.4f} Bal Acc: {:.4f} Auc: {:.4f}\n'.format(
                    phase, epoch_loss, curr_recall, curr_specificity, bal_acc, auc))

                if phase == 'val' and (auc > best_auc):
                    # deep copy the model
                    best_epoch = epoch
                    best_auc = auc
                    best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)

    return model, best_auc, best_epoch

def kFoldTraining(log_file, no_folds, writer, y, train_dataset, test_dataloader, test_indices, class_weights, hparams):
    '''
        By deafult 5-fold cross-validation training of a model, and after each fold test of a model.
        @param log_file: file for logs of training like train, val losses, metrics
        and confusion matrix for every epoch.
        @param no_folds: number of folds, by default it is 5-fold.
        @param writer: DictWriter object for result file.
        @param train_dataset: training dataset.
        @param test_dataloader: dataloader object for test dataset.
        @param test_indices: indices of test samples.
        @param class_weights: weights of classes.
        @param hparams: hyperparameters of a model.
    '''
    # Stratified K Fold (by default 5-fold)
    skf = StratifiedKFold()

    row = { "Learning_rate": hparams["lr"], "Dropout": hparams["Dropout"],
            "weight_decay": hparams["weight_decay"], "class_weight1": class_weights[0].item(),
            "class_weight2": class_weights[1].item(), "mixup_alpha": hparams["mixup_alpha"]}

    val_aucs = []
    test_aucs = []
    #fig, axes = plt.subplots(2, 3)
    #fig, axes = plt.subplots()
    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(y)), y)):
        log_file.write(f'FOLD {fold}\n')
        log_file.write('--------------------------------\n')
        # Get the model
        model = get_ResNet(hparams["Dropout"], no_layers=hparams["no_layers"], pretrained=hparams["pretrained"])
        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(class_weights.to(device))
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"],
                                    weight_decay = hparams["weight_decay"])
        # and SteLR
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hparams["step_size"],
                                                gamma=hparams["gamma"])

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_size = len(train_ids)
        val_size = len(val_ids)
        #val_size = len(test_indices)

        dataset_sizes = {'train': train_size, 'val': val_size}

        # Get dataloaders
        trainloader = DataLoader(train_dataset, batch_size=hparams["train_batch_size"], sampler=train_subsampler)
        validloader = DataLoader(train_dataset, batch_size=hparams["val_batch_size"], sampler=val_subsampler)

        dataloaders = {'train': trainloader, 'val': validloader}

        # train
        best_model, val_auc, model_epoch  = train_model(log_file,
                                                        model,
                                                        dataloaders,
                                                        dataset_sizes,
                                                        loss_fn,
                                                        optimizer,
                                                        exp_lr_scheduler,
                                                        num_epochs=hparams["num_of_epochs"], mixup=hparams["mixup"], alpha=hparams["mixup_alpha"])



        # test
        test_loss, test_auc = test(log_file, test_dataloader, len(test_indices), best_model, nn.CrossEntropyLoss())

        # append val and test bal_acc from current fold
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)


    for i in range(no_folds):
        new_val_key = "fold_" + str(i) + "_val_auc"
        row[new_val_key] = val_aucs[i]

    for i in range(no_folds):
        new_test_key = "fold_" + str(i) + "_test_auc"
        row[new_test_key] = test_aucs[i]

    row['mean_val_auc'] = np.mean(val_aucs)
    row['mean_test_auc'] = np.mean(test_aucs)
    # save the result row to the csv file
    writer.writerow(row)


def test(log_file, dataloader, size, model, criterion):
    '''
        Function for test model.
    '''
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # confusion matrix
    conf_mtx = {
        "TP" : 0,
        "FP" : 0,
        "TN" : 0,
        "FN" : 0
    }

    all_outputs = []
    all_preds = []
    all_labels = []
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            max_outputs = outputs[:, 1]
            loss = criterion(outputs, labels)
            calc_confusion_matrix(conf_mtx, preds, labels)

        all_outputs.extend(max_outputs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    all_outputs = np.array(all_outputs, dtype=float)
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    try:
        curr_recall = conf_mtx["TP"] / (conf_mtx["TP"] + conf_mtx["FN"])
    except ZeroDivisionError:
        curr_recall = 0

    try:
        curr_specificity = conf_mtx["TN"] / (conf_mtx["TN"] + conf_mtx["FP"])
    except ZeroDivisionError:
        curr_specificity = 0

    test_loss = running_loss / size
    test_bal_acc = (curr_recall + curr_specificity) / 2
    test_auc = roc_auc_score(all_labels, all_outputs, average='weighted')

    log_file.write('Test Confusion matrix: {}\n'.format(conf_mtx))
    log_file.write('Test Loss: {:.4f} Recall: {:.4f} Specificity: {:.4f} Bal Acc: {:.4f} Auc: {:.4f}\n'.format(
            test_loss, curr_recall, curr_specificity, test_bal_acc, test_auc))

    return test_loss, test_auc
