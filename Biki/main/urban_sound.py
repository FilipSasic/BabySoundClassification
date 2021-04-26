import setup

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

from processing.sound import Spectogram
from processing.dataset import SpectogramDataSet


if __name__ == '__main__':

    print('Urban Sound')

    # Params
    way_to_spectogram = False
    sample_way_to_spectogram = 500
    sample_spectogram_data = 500
    batch_size = 64

    wav_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/wav'
    img_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/img'
    meta_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/meta_data.csv'

    data_loader_valid_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/valid.pth'
    data_loader_train_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/train.pth'

    data_loader_valid_dataset_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/valid_dataset.pth'
    data_loader_train_dataset_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/train_dataset.pth'

    meta_data_predict_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data.csv'
    meta_data_predict_train_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data_train.csv'
    meta_data_predict_valid_path = '/workspaces/happiest_baby/happiest_baby/data/urbansound/loaders/meta_data_valid.csv'



    if way_to_spectogram == True:
        spectogram = Spectogram(sound_path=wav_path, specto_path=img_path)
        spectogram.transform(batch_size=1000, sample=sample_way_to_spectogram)

    # Prepare meta data
    meta_data = pd.read_csv(meta_path, dtype=str)
    meta_data['ID'] = meta_data['ID'].astype(int)
    meta_data = meta_data.sort_values(by=['ID'])

    if sample_spectogram_data != None:
        meta_data = meta_data[: sample_spectogram_data]
    meta_data['ClassID'] = pd.factorize(meta_data.Class)[0]

    # Train Valid Split 
    meta_data_train, meta_data_valid = train_test_split(meta_data, random_state=42, test_size=0.2)
    meta_data.to_csv(meta_data_predict_path, index=False)
    meta_data_train.to_csv(meta_data_predict_train_path, index=False)
    meta_data_valid.to_csv(meta_data_predict_valid_path, index=False)

    train_dataset = SpectogramDataSet(dataframe=meta_data_train, img_path=img_path)
    valid_dataset = SpectogramDataSet(dataframe=meta_data_valid, img_path=img_path)

    # DataLoader
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Save Data
    torch.save(dataloader_train, data_loader_train_path)
    torch.save(dataloader_valid, data_loader_valid_path)


    # DataForMetrics
    train_dataset = SpectogramDataSet(dataframe=meta_data_train, img_path=img_path, predict=True)
    valid_dataset = SpectogramDataSet(dataframe=meta_data_valid, img_path=img_path, predict=True)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    torch.save(dataloader_train, data_loader_train_dataset_path)
    torch.save(dataloader_valid, data_loader_valid_dataset_path)



