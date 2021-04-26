
import gc
import os
import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt
from matplotlib import figure

from tqdm import trange

class Spectogram:

    def __init__(self, sound_path, specto_path, *params, **kwargs):
        
        
        self.sound_path = sound_path

        if specto_path[-1] == '/':
            self.specto_path = specto_path
        else:
            self.specto_path = f'{specto_path}/'


    def transform(self, batch_size=1000, progress_bar=True, sample=None):

        root, dirs, files = next(os.walk(self.sound_path))
        files.sort(key=lambda x: int(x[: -4]))
        
        if sample != None:
            files = files[: sample]
            
        
        with trange(len(files)) as num_of_files:

            for file_index in num_of_files:
                file = files[file_index: file_index+1][0]

                if progress_bar == True:
                    num_of_files.set_description(f'Progress: wav to spectogram')

                name = file.split('.')[0]
                wav_path = root + '/' + file

                self.create_spectrogram(filename=wav_path, name=name, specto_path=self.specto_path)

                if file_index % batch_size == 0:
                    gc.collect()
            
        gc.collect()
        

    def create_spectrogram(self, filename, name, specto_path):

        plt.interactive(False)
        clip, sample_rate = librosa.load(filename, sr=None)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        filename  = specto_path + name + '.jpg'
        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()    
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del filename,name,clip,sample_rate,fig,ax,S
