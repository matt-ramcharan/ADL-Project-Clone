import librosa
import numpy as np
import pandas as pd
import pickle


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


def augment(data):
    return pd.DataFrame(np.concatenate([[[time_stretch(row.get("data"), 0.5), row.get("labels"), row.get("track_id")],
                                         [time_stretch(row.get("data"), 0.2), row.get("labels"), row.get("track_id")],
                                         [time_stretch(row.get("data"), 1.2), row.get("labels"), row.get("track_id")],
                                         [time_stretch(row.get("data"), 1.5), row.get("labels"), row.get("track_id")],
                                         [pitch_shift(row.get("data"), -2), row.get("labels"), row.get("track_id")],
                                         [pitch_shift(row.get("data"), -5), row.get("labels"), row.get("track_id")],
                                         [pitch_shift(row.get("data"), 2), row.get("labels"), row.get("track_id")],
                                         [pitch_shift(row.get("data"), 5), row.get("labels"), row.get("track_id")],
                                         [row.get("data"), row.get("labels"), row.get("track_id")]]
                                        for idx, row in data.iterrows()], axis=0),
                        columns=["data", "labels", "track_id"])


def save_augmented():
    pickle_in = open("music_genres_dataset.pkl", "rb")
    print("Stage 1 Complete")
    dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))
    print("Stage 2 Complete")
    dataset_new = augment(dataset)
    print("Stage 3 Complete")
    pickle.dump(dataset_new, open("music_genres_dataset_aug.pkl", "wb"))
    print("Stage 4 Complete")


if __name__ == "__main__":
    save_augmented()