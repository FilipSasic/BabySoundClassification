import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

def create_spectocram_image(audio_file):
    clip, sample_rate = librosa.load(audio_file, sr=None)

    # D = librosa.stft(clip)
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    M = librosa.feature.melspectrogram(y=clip, sr=sample_rate, hop_length=256, n_fft=4096)
    M_db = librosa.power_to_db(M, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(M_db,hop_length=256,x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.savefig("spectogram.png")

