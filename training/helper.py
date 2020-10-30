import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

num_classes = None


class Loader:
    dataset = None
    dshape = None

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        d = self.dshape
        instrs = np.random.randint(d[0], size=self.batch_size)
        tracks = np.random.randint(d[1], size=self.batch_size)
        starts = np.random.randint(d[-1] - 178*10, size=self.batch_size)
        xs = np.zeros((self.batch_size, 1, 108, 178))
        for i in range(self.batch_size):
            xs[i] = np.copy(self.dataset[instrs[i], tracks[i], ..., starts[i] + 178*5: starts[i] + 178*6].reshape(1, 108, 178))
        ys = instrs
        return torch.from_numpy(xs), torch.from_numpy(ys)


class TrackLoader:
    def __init__(self, chromo=None, maxlength=None):
        self.chromo = chromo
        self.maxlength = maxlength

    def second(self, i):
        assert self.maxlength > i, "only maxlength seconds allowed"
        xs = np.copy(self.chromo[..., 178*i: 178*(i+1)].reshape(1, 1, 108, 178))
        return torch.from_numpy(xs)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes, batch_size):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 80))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(1, batch_size, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def add_pr_curve_tensorboard(class_index, test_probs, test_preds, classes, writer, global_step=0):
    """
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()
