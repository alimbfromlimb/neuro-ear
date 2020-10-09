import numpy as np
import librosa, librosa.display
import glob


def to_files(instruments):
    """
    :return: a list of lists (for each instrument) of file_names.wav
    """
    n_inst = len(instruments)
    pathss = []
    for i in range(n_inst):
        pathss.append(glob.glob(instruments[i] + '/*.wav'))

    return pathss


def file_to_chromagram(file_name):
    sr = 44100
    x, sr = librosa.load(file_name, sr=sr)  # .wav file and its sampling rate
    fmin = librosa.midi_to_hz(22)  # minimal key on our chromagram will be A0
    hop_length = 256  # needed for Constant-Q Transform
    amplitude = librosa.cqt(x[:120*44100], sr=sr, fmin=fmin, n_bins=108, hop_length=hop_length)
    chromagram = librosa.amplitude_to_db(np.abs(amplitude))
    return chromagram


instruments = ["accordion", "guitar", "piano", "violin"]
n_inst = len(instruments)
files_per_inst = 15


# making arrays of values (xs) and targets (ys)
# creating xs first

xs = np.zeros((n_inst, files_per_inst, 108, 20672))
pathss = to_files(instruments)

for i in range(n_inst):
    print('   ', i)
    for j in range(files_per_inst):
        print(j)
        file = pathss[i][j]
        chromagram = file_to_chromagram(file)
        assert chromagram.shape[0] == 108, (chromagram.shape, file)
        xs[i, j, ...] = np.copy(chromagram)

# now creating ys
ys = np.eye(5, dtype=int)


np.save('xs.npy', xs)
np.save('ys.npy', ys)