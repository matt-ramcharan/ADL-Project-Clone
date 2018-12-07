import librosa
import numpy as np


def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


def time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate)


def pitch_shift(audio, shift_steps):
    return librosa.effects.pitch_shift(audio, sr, n_steps=shift_steps)


def augment(data):
    ts_05 = time_stretch(data, 0.5)
    ts_02 = time_stretch(data, 0.2)
    ts_15 = time_stretch(data, 1.5)
    ts_12 = time_stretch(data, 1.2)
    ps_2 = pitch_shift(data, -2)
    ps_5 = pitch_shift(data, -5)
    ps2 = pitch_shift(data, 2)
    ps5 = pitch_shift(data, 5)
