'''
A dictionary which contains hyperparameters for given type of CNN.
'''
hparams = {
                # ResNet34 hparams
                "ResNet34" : {
                    "Dropout": 0.2,
                    "lr" : 5e-3,
                    "weight_decay" : 0.0005,
                    "num_of_epochs" : 100,
                    "train_batch_size": 64,
                    "val_batch_size" : 64,
                    "step_size" : 7,
                    "gamma" : 0.1,
                    "no_layers" : 34,
                    "pretrained": False,
                    "mixup": False,
                    "mixup_alpha": 0.4
                },
                # ResNet18 hparams
                "ResNet18" : {
                    "Dropout": 0.2,
                    "lr" : 1e-1,
                    "weight_decay" : 0.0001,
                    "num_of_epochs" : 100,
                    "train_batch_size": 64,
                    "val_batch_size" : 64,
                    "step_size" : 7,
                    "gamma" : 0.1,
                    "no_layers" : 18,
                    "pretrained": False,
                    "mixup": False,
                    "mixup_alpha": 0.4
                }
}
