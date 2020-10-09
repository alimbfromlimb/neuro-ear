import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from helper import plot_classes_preds, add_pr_curve_tensorboard
from helper import Loader
from architecture import *
# from torch.optim.lr_scheduler import MultiplicativeLR


exp_name = 'adam_0'
writer = SummaryWriter('runs/'+exp_name)


# Hyperparameters
num_epochs = 20
batch_size = 30
batches_per_epoch = 500
learning_rate = 1e-4
MODEL_STORE_PATH = 'models/'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
classes = ('accordion', 'guitar', 'piano', 'violin')

# DATA_PATH = '/Users/bukharaevalimniazovich/Desktop/neuro-ear_datasets/'
DATA_PATH = ''

data_train = np.load(DATA_PATH + 'train.npy')
data_test = np.load(DATA_PATH + 'test.npy')
data_validation = np.load(DATA_PATH + 'validation.npy')


class TrainLoader(Loader):
    dataset = data_train
    dshape = dataset.shape


class ValidLoader(Loader):
    dataset = data_validation
    dshape = dataset.shape


class TestLoader(Loader):
    dataset = data_test
    dshape = dataset.shape


trainloader = TrainLoader(batch_size)
testloader = TestLoader(batch_size)
validloader = ValidLoader(batch_size)

net = ConvNet()

# def init_all(model, init_func, *params, **kwargs):
#     for p in model.parameters():
#         init_func(p, *params, **kwargs)

# init_all(net, torch.nn.init.normal_, mean=0., std=1)

# net.load_state_dict(...)

# Loss and optimizer
lr = 1e-4
criterion = nn.BCEWithLogitsLoss(reduce='mean') # nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# lmbda = lambda epoch: 0.95
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)


loss_list = []
acc_list = []
running_loss = 0.0
for epoch in range(num_epochs):
    print(epoch)
    net.train()
    for i in range(batches_per_epoch):
        images, labels = next(trainloader)
        images, labels = images.to(torch.float), labels.to(torch.long)
        ys = torch.zeros((batch_size, num_classes))
        ys[np.arange(batch_size), labels] = 1
        # Run the forward pass
        outputs = net(images)
        loss = criterion(outputs, ys)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            if epoch == 0 and i == 0:
                running_loss = 0.0
                continue
            print(' ', i)
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * batches_per_epoch + i)

            # ...log a Matplotlib Figure showing the net's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, images, labels, classes, batch_size),
                            global_step=epoch * batches_per_epoch + i)
            running_loss = 0.0
    # scheduler.step()

    class_probs = []
    class_preds = []
    net.eval()
    with torch.no_grad():
        for data in range(20):
            images, labels = next(validloader)
            images, labels = images.to(torch.float), labels.to(torch.long)
            outputs = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            _, class_preds_batch = torch.max(outputs, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    val_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    val_preds = torch.cat(class_preds)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, val_probs, val_preds, classes, writer)


writer_ = SummaryWriter('runs/test/'+exp_name)
class_probs = []
class_preds = []
net.eval()
with torch.no_grad():
    for data in range(500):
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
    add_pr_curve_tensorboard(i, test_probs, test_preds, classes, writer_)

torch.save(net.state_dict(), MODEL_STORE_PATH + exp_name+'.ckpt')