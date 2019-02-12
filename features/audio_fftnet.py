import librosa
import librosa.filters
import numpy as np
import pysptk
import pyworld
from scipy import signal
from scipy.interpolate import interp1d

"""Reference:
    https://github.com/keithito/tacotron/blob/master/util/audio.py
    https://github.com/kan-bayashi/PytorchWaveNetVocoder/blob/master/src/bin/feature_extract.py
"""


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.int16), sample_rate)


def preemphasis(x, preemph):
    return signal.lfilter([1, -preemph], [1], x)


def inv_preemphasis(x, preemph):
    return signal.lfilter([1], [1, -preemph], x)


def melspectrogram(y, preemph, ref_level_db, min_level_db, num_freq, frame_shift_ms,
                   frame_length_ms, sample_rate, num_mels):
    D = _stft(preemphasis(y, preemph), num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    S = _amp_to_db(_linear_to_mel(np.abs(D), num_freq, sample_rate, num_mels)) - ref_level_db
    return _normalize(S, min_level_db).T


def _stft(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def extract_mcc(wav, num_freq, frame_shift_ms, sample_rate, minf0, maxf0, mcep_dim, mcep_alpha):
    wav = np.array(wav, dtype=np.float)
    n_fft = (num_freq - 1) * 2
    f0, time_axis = pyworld.harvest(wav, sample_rate, f0_floor=minf0,
                                    f0_ceil=maxf0, frame_period=frame_shift_ms)
    spc = pyworld.cheaptrick(wav, f0, time_axis, sample_rate, fft_size=n_fft)

    f0[f0 < 0] = 0
    uv, cont_f0 = convert_continuos_f0(f0)

    mcep = pysptk.sp2mc(spc, mcep_dim, alpha=mcep_alpha)
    uv = np.expand_dims(uv, axis=-1)
    cont_f0 = np.expand_dims(cont_f0, axis=-1)
    feats = np.concatenate([uv, cont_f0, mcep], axis=1)
    return feats


def convert_continuos_f0(f0):
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


# Conversions:

_mel_basis = None


def _linear_to_mel(spectrogram, num_freq, sample_rate, num_mels):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(num_freq, sample_rate, num_mels)
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis(num_freq, sample_rate, num_mels):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)
