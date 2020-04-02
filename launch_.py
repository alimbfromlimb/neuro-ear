import numpy as np
import wave
import os
import sys
import math
# here I am blocking "Using tensorflow backend" message from printing to a terminal
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.io import wavfile

# Again, getting rid of logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
This file launches a neural net on a given wav audio file.
The net predicts an instrument that is most likely to be played on the audio
and outputs the corresponding number to a text file (please see below).
It also outputs mean probabilities of each instrument.

0 -- accordion
1 -- piano
2 -- violin
3 -- guitar
4 -- noise
5 -- error

To use it, please:
    1) Make sure you have model.h5, launch.py and an audio file you would like
    to work with placed in one directory (for example, .../Instrument_classifier).
    2) Open a terminal and go to that directory.
    3) Then use this command:
    python -W ignore launch.py your_file_name.wav
    4) See the results in answer.txt file in the same directory
'''

def get_wav_info(wav_filename):
    """
    Loads a wav file.
    Input: file name.
    Returns: rate of the audio, raw data itself
    """
    rate, data = wavfile.read(wav_filename)
    return rate, data


def graph_spectrogram(rate, data):
    """
    Calculate and plot spectrogram for a wav audio file.
    Input: rate of the audio and raw data itself.
    Returns: 2-dimensional ndarray (a spectrogram).
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


def get_duration_wav(wav_filename):
    """
    Gets duration of an audio file.
    Input: file name.
    Returns: its duration.
    """
    f = wave.open(wav_filename, 'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    return duration


def most_frequent(List):
    """
    Input: a list.
    Returns: its most frequent element.
    """
    return max(set(List), key=List.count)


def what_is_it(Y):
    """
    Input: neural net's predictions.
    Returns: number of the predicted instrument.
    """
    list_ = []
    for row in Y:
        list_.append(np.argmax(row))
    res = most_frequent(list_)
    return int(res)


def round_half_up(n, decimals=0):
    """
    Rounds a number.
    """
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def inst_probability(Y):
    """
    Input: neural net's predictions.
    Returns: mean probabilities of each instrument to present in this audio.
    """
    Y = Y.T
    res = []
    num = Y.shape[1]
    for row in Y:
        res.append(round_half_up(np.sum(row)/num, 3)*100)
    return res


def instrument_classifier(file, model, window = 1.0):

    # Let's define the "window size" of an audioclip which then will be turned into a spectrogram
    dur = get_duration_wav(file)

    # Spectrograms for each clip will be stored here.
    X = []

    rate, data = get_wav_info(file)
    if data.ndim > 1:
        data = np.delete(data, 1, 1)  # Leaving only one channel.

    # a = int((1 // window) * 10)
    # b = int((1 // window) * dur) - 10

    a = 0
    b = int(dur // window)

    for s in range(a, b, 2):
        data_ = data
        t1 = s * window
        t2 = t1 + window
        t1 = int(t1 * 44100)  # Works in milliseconds.
        t2 = int(t2 * 44100)
        data_ = data_[t1:t2]
        x = graph_spectrogram(rate, data_)
        X.append(x)

    X = np.array(X)
    maxElement = 16462066.603519555
    X = X / maxElement  # Normalizing data.

    train_shape = list(X.shape)
    train_shape.append(1)
    X = X.reshape(train_shape)

    Y = model.predict(X)

    # Printing results to answer.txt.

    inst = what_is_it(Y)
    res = inst_probability(Y)
    str_res = str(inst) + ':       ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + ' ' + str(res[3]) + ' ' + str(res[4])
    return inst

