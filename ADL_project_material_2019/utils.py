import librosa
import numpy as np
import pandas as pd
import pickle
import multiprocessing

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
