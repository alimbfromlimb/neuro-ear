import librosa, librosa.display

import numpy as np
import matplotlib
import torch
import torch.nn.functional as F
from training.helper import TrackLoader


def file_to_chromagram(file_name):
    sr = 44100
    x, sr = librosa.load(file_name, sr=sr)  # .wav file and its sampling rate
    fmin = librosa.midi_to_hz(22)  # minimal key on our chromagram will be A0
    hop_length = 256  # needed for Constant-Q Transform
    amplitude = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=108, hop_length=hop_length)
    chromagram = librosa.amplitude_to_db(np.abs(amplitude))
    l = x.shape[-1] // sr
    return chromagram, l


def classifier_torch(file_name, net, maxlength_=None):

    chroma, tracklength = file_to_chromagram(file_name)

    if maxlength_ is None:
        maxlength_ = tracklength

    instruments = ["accordion", "guitar", "piano", "violin"]
    n_inst = len(instruments)

    class Tmp(TrackLoader):
        chromo = chroma
        maxlength = maxlength_

    trackloader = Tmp()

    class_probs = []
    net.eval()
    with torch.no_grad():
        for data in range(min(maxlength_, tracklength)):
            images = trackloader.second(data)
            images = images.to(torch.float)
            outputs = net(images)
            class_probs_batch = [F.sigmoid(el) for el in outputs]
            class_probs.append(class_probs_batch)

    test_probs = np.array(torch.cat([torch.stack(batch) for batch in class_probs]))[:, :n_inst]

    im = test_probs.T
    padding = np.zeros((4, 25))
    probs = np.repeat(padding + im[:, :25], 150, axis=0)
    probs = np.repeat(probs, 50, axis=1)
    matplotlib.image.imsave('static/probs.png', 100 - probs, cmap='Blues')
    print("saved")

