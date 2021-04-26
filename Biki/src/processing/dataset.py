import torch
import torchvision
from PIL import Image

class SpectogramDataSet(torch.utils.data.Dataset):


    def __init__(self, dataframe, img_path, img_size=(64, 64), predict=False):
        self.dataframe = dataframe
        self.path = img_path
        self.img_size = img_size
        self.predict = predict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]
        im = Image.open(f'{self.path}/{row["ID"]}.jpg')
        im.thumbnail(self.img_size)

        if self.predict == False:

            return (
                torchvision.transforms.functional.to_tensor(im),
                row["ClassID"],
            )

        else:
            return torchvision.transforms.functional.to_tensor(im)