import librosa
import librosa.display
import numpy as np
import pandas as pd
import pickle
import multiprocessing
import matplotlib.pyplot as plt

def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def time_stretch(audio, rate):
    stretched = librosa.effects.time_stretch(audio, rate)
    if rate < 1:
        return stretched[0:len(audio)]
    else:
        x = np.zeros_like(audio)
        x[0:len(stretched)] = stretched
        return x


def pitch_shift(audio, shift_steps):
    return librosa.effects.pitch_shift(audio, 22050, n_steps=shift_steps, bins_per_octave=12)

def augment_segment(x):
    stretches = list(map(lambda y:[time_stretch(x[0], y)] + [x[1], x[2]], [0.5, 0.2, 1.2, 1.5]))
    shifts = list(map(lambda y:list(map(lambda z:[pitch_shift(x[0], y)] + [x[1], x[2]], stretches)), [-5, -2, 2, 5]))
    shifts = list(np.array(shifts).reshape((16, 3)))
    return shifts

def augment_song(x):
    print(x[0][2])
    x = list(x)
    indices = np.random.randint(15, size=3)
    for index in indices:
        x = x + augment_segment(x[index])
    return x

def augment(data):
    pool = multiprocessing.Pool()
    intial_data = np.row_stack(data.values)
    songs = np.split(intial_data, 750)
    new_songs = np.array(list(pool.map(augment_song, songs)))
    print(new_songs.shape)
    new_songs = new_songs.reshape((new_songs.shape[0] * new_songs.shape[1], 3))
    df = pd.DataFrame(new_songs, columns=['data', 'labels', 'track_id'])
    print(df)
    return df

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def spectrogram_plot(groups,classes):
    track_label = groups['labels'].iloc[0]
    track_class = classes[track_label]
    track_id = groups['track_id'].iloc[0]
    i=0

    for tseries in groups['data']:
        plt.figure(figsize=(12, 8))
        plt.title("track_id")
        plt.subplot(2, 1, 1)
        librosa.display.waveplot(tseries, sr=22050)
        plt.title('amplitude envelope of audio waveform')

        mel = melspectrogram(tseries)
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(mel ** 2, np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')
        dir = 'spectrograms/' + track_class + '/ID_' + str(track_id) + '_' + '/'
        mkdir_p(dir)
        plt.savefig(dir + str(i) +'.pdf', format='pdf')
        i+=1

def save_specto_plots():
    classes = { 0:"blues",
                1:"classical",
                2:"country",
                3:"disco",
                4:"hiphop",
                5:"jazz",
                6:"metal",
                7:"pop",
                8:"reggae",
                9:"rock",
                }
    pickle_in = open("music_genres_dataset.pkl", "rb")
    print("Stage 1 Complete")
    dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))
    print("Stage 2 Complete")

    groups = [data for _, data in dataset.groupby('track_id')]

    for i in groups:
        spectrogram_plot(i,classes)

    print("Stage 3 Complete")

def save_augmented():
    pickle_in = open("music_genres_dataset.pkl", "rb")
    print("Stage 1 Complete")
    dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))
    print("Stage 2 Complete")
    dataset_new = augment(dataset)
    print("Stage 3 Complete")
    pickle.dump(dataset_new, open("music_genres_dataset_aug.pkl", "wb"), protocol=2)
    print("Stage 4 Complete")


if __name__ == "__main__":
    save_augmented()
