import torch
from torch import nn
import numpy as np

class TorchModel(nn.Module):

    def __init__(self, *params, **kwargs):
        super(TorchModel, self).__init__()


    def fit(self, criterion, optimizer, dataloader_train=None, dataloader_test=None, n_epochs=100, progress_bar=True, device='cpu',*args, **kwargs):

        from datetime import datetime

        train_losses = []
        test_losses = []

        for it in range(n_epochs):
            t0 = datetime.now()
            train_loss = []

            for inputs, targets in dataloader_train:

                # to device
                inputs, targets = inputs.to(device), targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            # Get train loss and test loss
            train_loss = np.mean(train_loss)

            test_loss = []
            for inputs, targets in dataloader_test:

                # to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

            # Save losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            dt = datetime.now() - t0
            print(f'Epoch {it+1}/{n_epochs}, Train Loss: {train_loss:.4f}, '
                    f'Test Loss: {test_loss:.4f}, Duration: {dt}')

        return train_losses, test_losses


    def predict(self, dataloader, device='cpu'):

        outputs = []
        for data in dataloader:

            if len(data) == 2:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = data.to(device)


            output_chunk = self.model(inputs)
            output_chunk = to_numpy(output_chunk)
            
            outputs.append(output_chunk)

        outputs = np.concatenate(outputs)
        return outputs

class Ann:

    def __init__(self, input_size, output_size, num_of_layer=1, step=2, *args, **kwargs):

        lst_1 = nn.ModuleList()
        out_size = input_size


        for i in range(num_of_layer):
            inp_size = out_size
            out_size = int(inp_size // step)
            
            if out_size < output_size:
                break

            if i == num_of_layer-1:
                block = nn.Sequential(nn.Linear(inp_size, output_size), nn.Softmax())
            else:
                block = nn.Sequential(nn.Linear(inp_size, out_size), nn.BatchNorm1d(num_features = out_size), nn.ReLU())
            lst_1.append(block)
        
        self.model = nn.Sequential(*lst_1)



from sklearn.metrics import f1_score    
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

class Metrics:


    def __init__(self, *args, **kwargs):
        pass

    def accuracy(self, y_true, y_predict):

        y_true = np.round(to_numpy(y_true))
        y_predict = np.round(to_numpy(y_predict))

        acc = np.mean(y_true == y_predict)

        return acc

    def f1(self, y_true, y_predict):

        y_true = np.round(to_numpy(y_true))
        y_predict = np.round(to_numpy(y_predict))

        f1_macro = f1_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='macro')
        f1_micro = f1_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='micro')

        return f1_macro, f1_micro

    
    def precision(self, y_true, y_predict):

        y_true = np.round(to_numpy(y_true))
        y_predict = np.round(to_numpy(y_predict))

        precision_score_macro = precision_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='macro')
        precision_score_micro = precision_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='micro')

        return precision_score_macro, precision_score_micro


    def recall(self, y_true, y_predict):

        y_true = np.round(to_numpy(y_true))
        y_predict = np.round(to_numpy(y_predict))

        recall_score_macro = recall_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='macro')
        recall_score_micro = recall_score(y_true=y_true, y_pred=y_predict.argmax(axis=1), average='micro')

        return recall_score_macro, recall_score_micro

    def auc(self, y_true, y_predict):

        y_true = np.round(to_numpy(y_true))
        y_predict = np.round(to_numpy(y_predict))

        auc = roc_auc_score(y_true=y_true, y_score=y_predict, multi_class='ovr')

        return auc



def to_tensor(array):

    if isinstance(array, np.ndarray):
        if len(array.shape) == 1:
            array = array.reshape(array.shape[0], 1)
        return torch.from_numpy(array.astype(np.float32))
    else:
        if len(array.shape) == 1:
            array = array.reshape(array.shape[0], 1)

        array = array.type(torch.float32)
    return array

def to_numpy(tensor):
    
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().numpy()
        if len(array.shape) == 1:
            array = array.reshape(array.shape[0], 1)
        return array
    else:
        if len(tensor.shape) == 1:
            tensor = tensor.reshape(tensor.shape[0], 1)
        return tensor