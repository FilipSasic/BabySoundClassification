import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

def create_spectocram_image(audio_file):
    clip, sample_rate = librosa.load(audio_file, sr=None)

    D = librosa.stft(clip)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(S_db, ax=ax)
    # fig.colorbar(img, ax=ax)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Using a logarithmic frequency axis')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig("spectogram.png")

