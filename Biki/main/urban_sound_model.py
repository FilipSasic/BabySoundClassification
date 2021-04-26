import setup

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from models.classifiers import SoudImgClassifier

if __name__ == '__main__':

    # Model params
    lr = 0.0001
    n_epochs = 350
    output_size = 10 # num of classes
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    model_name = 'resnet18'

    # Ann params
    ann_input_size = 512
    ann_num_of_layer = 6 
    ann_step = 2

    ann_params = {
        'ann_input_size': ann_input_size,
        'ann_num_of_layer': ann_num_of_layer,
        'ann_step': ann_step
    }


    # Create ML model
    model = SoudImgClassifier(ann_input_size=ann_input_size, ann_outputsize=10, ann_num_of_layer=ann_num_of_layer, ann_step=ann_step)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Get Data
    data_loader_valid_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/valid.pth'
    data_loader_train_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/train.pth'
    data_loader_valid_dataset_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/valid_dataset.pth'
    data_loader_train_dataset_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/train_dataset.pth'
    meta_data_predict_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data.csv'
    meta_data_predict_train_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data_train.csv'
    meta_data_predict_valid_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data_valid.csv'
    logs_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/logs.csv'
    losses_fig_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/img'


    try:
        log_df = pd.read_csv(logs_path)
    except:
        log_columns = ['model_name', 'lr', 'n_epochs', 'ann_params', 'train_loss', 'valid_loss',
                        'accuracy_train', 'accuracy_valid', 'f1_macro_train', 'f1_micro_train',
                        'f1_macro_valid', 'f1_micro_valid', 'precision_macro_train', 'precision_micro_train',
                        'precision_macro_valid', 'precision_micro_valid', 'recall_macro_train', 'recall_micro_train',
                        'recall_macro_valid', 'recall_micro_valid']
        log_df = pd.DataFrame(columns=log_columns)

    dataloader_train = torch.load(data_loader_train_path)
    dataloader_valid = torch.load(data_loader_valid_path)
    
    model.to(device)
    train_losses, valid_losses = model.fit(optimizer=optimizer, criterion=criterion, 
            dataloader_train=dataloader_train, dataloader_test=dataloader_valid, 
            n_epochs=n_epochs, device=device)

    train_loss = min(train_losses)
    valid_loss = min(valid_losses)
    
    # Data For predict
    dataset_train = torch.load(data_loader_train_dataset_path)
    dataset_valid = torch.load(data_loader_valid_dataset_path)
    meta_data = pd.read_csv(meta_data_predict_path)
    meta_data_train = pd.read_csv(meta_data_predict_train_path)
    meta_data_valid = pd.read_csv(meta_data_predict_valid_path)

    y_true_train = meta_data_train['ClassID'].values
    y_true_valid = meta_data_valid['ClassID'].values

    outputs_train = model.predict(dataloader=dataset_train)
    outputs_valid = model.predict(dataloader=dataset_valid)

    # Metrics
    accuracy_train = model.accuracy(y_true=y_true_train, y_predict=outputs_train)
    accuracy_valid = model.accuracy(y_true=y_true_valid, y_predict=outputs_valid)

    f1_macro_train, f1_micro_train = model.f1(y_true=y_true_train, y_predict=outputs_train)
    f1_macro_valid, f1_micro_valid = model.f1(y_true=y_true_valid, y_predict=outputs_valid)

    precision_macro_train, precision_micro_train = model.precision(y_true=y_true_train, y_predict=outputs_train)
    precision_macro_valid, precision_micro_valid = model.precision(y_true=y_true_valid, y_predict=outputs_valid)

    recall_macro_train, recall_micro_train = model.recall(y_true=y_true_train, y_predict=outputs_train)
    recall_macro_valid, recall_micro_valid = model.recall(y_true=y_true_valid, y_predict=outputs_valid)
    

    
    row = [model_name, lr, n_epochs, ann_params, train_loss, valid_loss,
            accuracy_train, accuracy_valid, f1_macro_train, f1_micro_train, f1_macro_valid, f1_micro_valid,
            precision_macro_train, precision_micro_train, precision_macro_valid, precision_micro_valid,
            recall_macro_train, recall_micro_train, recall_macro_valid, recall_micro_valid]

    log_df.loc[len(log_df)] = row
    log_df.to_csv(logs_path, index=False)
 
    print(log_df.head())


    plt.plot(train_losses, label='Train loss')
    plt.plot(valid_losses, label='Valid loss')
    plt.legend()
    plt.savefig(f'{losses_fig_path}/{len(log_df)}.png')
    # plt.show()