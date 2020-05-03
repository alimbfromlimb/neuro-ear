import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from helper import matplotlib_imshow, images_to_probs, plot_classes_preds, add_pr_curve_tensorboard
from helper import Loader
from architecture import ConvNet

# Hyperparameters
num_epochs = 1000
num_classes = 10
batch_size = 6
batches_per_epoch = 1000
learning_rate = 1e-4

# DATA_PATH = 'C:\\Users\Andy\PycharmProjects\MNISTData'
NET_STORE_PATH = 'Desktop/neuro-ear/training/'
Data = np.load('/Users/bukharaevalimniazovich/Desktop/neuro-ear_datasets/youtube_12_keys/xs.npy')
data_train = Data[:, :15, ...]
data_test = Data[:, 15:, ...]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

classes = ('accordion', 'guitar', 'piano', 'violin')
writer = SummaryWriter('runs/test')


class TrainLoader(Loader):
    dataset = data_train
    dshape = dataset.shape


class TestLoader(Loader):
    dataset = data_test
    dshape = dataset.shape


trainloader = TrainLoader(batch_size)
testloader = TestLoader(batch_size)


net = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_list = []
acc_list = []
running_loss = 0.0
for epoch in range(num_epochs):
    for i in range(batches_per_epoch):
        images, labels = next(trainloader)
        images, labels = images.to(torch.float), labels.to(torch.long)
        # Run the forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        # acc_list.append(correct / total)

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #       .format(epoch + 1, num_epochs, i + 1, batches_per_epoch, loss.item(),
            #               (correct / total) * 100))

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the net's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, images, labels, classes),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0

class_probs = []
class_preds = []
net.eval()
with torch.no_grad():
    for data in range(1000):
        images, labels = next(testloader)
        images, labels = images.to(torch.float), labels.to(torch.long)
        outputs = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
        _, class_preds_batch = torch.max(outputs, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds, classes, writer)


# Test the net
# net.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     N = 100
#     for i in range(N):
#         images, labels = next(testloader)
#         images, labels = images.to(torch.float), labels.to(torch.long)
#         # Run the forward pass
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the net on the {} * {} test images: {} %'.format(batch_size, N, (correct / total) * 100))

# Save the net and plot
torch.save(net.state_dict(), NET_STORE_PATH + 'conv_net.ckpt')
