
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr

# Models
from model import FresUNet

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
import time
from itertools import chain
import time
import warnings
from pprint import pprint

from data import ChangeDetectionDataset, RandomFlip, RandomRot



def test(loader, criterion, net):

    net.eval()
    tot_loss = 0
    tot_count = 0

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_accuracy = list(0. for i in range(2))
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for batch in loader:
        I1 = batch['I1'].float().cuda()
        I2 = batch['I2'].float().cuda()
        label = torch.squeeze(batch['label'].cuda())

        output = net(I1, I2)

        loss = criterion(output, label.long())
        tot_loss += loss.data * np.prod(label.size())
        _, predicted = torch.max(output.data, 1)
        c = (predicted.int() == label.data.int())
        tot_count += np.prod(label.size())

        for i in range(c.size(1)):
            for j in range(c.size(2)):
                l = int(label.data[0, i, j])
                class_correct[l] += c[0, i, j]
                class_total[l] += 1

        pr = (predicted.int() > 0).cpu().numpy()
        gt = (label.data.int() > 0).cpu().numpy()

        tp += np.logical_and(pr, gt).sum()
        tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp += np.logical_and(pr, np.logical_not(gt)).sum()
        fn += np.logical_and(np.logical_not(pr), gt).sum()

    net_loss = tot_loss / tot_count
    net_accuracy = 100 * (tp + tn) / tot_count
    for i in range(2):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

    return net_loss, net_accuracy, class_accuracy, pr_rec


def train(n_epochs, net, criterion,  save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_change_accuracy = 0 * t
    epoch_train_nochange_accuracy = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t

    best_fm = 0

    best_lss = 1000

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        # Train loop
        for batch in train_loader:
            I1 = batch['I1'].float().cuda()
            I2 = batch['I2'].float().cuda()
            label = torch.squeeze(batch['label'].cuda())

            optimizer.zero_grad()
            output = net(I1, I2)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Eval
        epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec = test(train_loader, criterion, net)
        epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_train_change_accuracy[epoch_index] = cl_acc[1]
        epoch_train_precision[epoch_index] = pr_rec[0]
        epoch_train_recall[epoch_index] = pr_rec[1]
        epoch_train_Fmeasure[epoch_index] = pr_rec[2]

        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_loader, criterion, net)
        epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_test_change_accuracy[epoch_index] = cl_acc[1]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]

        fm = epoch_train_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            save_str = 'net-best.pth.tar'
            torch.save(net.state_dict(), save_str)

        lss = epoch_train_loss[epoch_index]
        if lss < best_lss:
            best_lss = lss
            save_str = 'net-best.pth.tar'
            torch.save(net.state_dict(), save_str)

        print("train loss:", epoch_train_loss[epoch_index],
              "train nochange acc:", epoch_train_nochange_accuracy[epoch_index],
              "train change acc:", epoch_train_change_accuracy[epoch_index])

        print("test loss:", epoch_test_loss[epoch_index],
              "test nochange acc:", epoch_test_nochange_accuracy[epoch_index],
              "test change acc:", epoch_test_change_accuracy[epoch_index])


    out = {'train_loss': epoch_train_loss[-1],
           'train_accuracy': epoch_train_accuracy[-1],
           'train_nochange_accuracy': epoch_train_nochange_accuracy[-1],
           'train_change_accuracy': epoch_train_change_accuracy[-1],
           'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1],
           'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
           'test_change_accuracy': epoch_test_change_accuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    print(pr_rec)

    return out


def save_test_results(dset):
    for name in tqdm(dset.names):
        with warnings.catch_warnings():
            I1, I2, cm = dset.get_img(name)
            I1 = torch.unsqueeze(I1, 0).float().cuda()
            I2 = torch.unsqueeze(I2, 0).float().cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
            io.imsave(f'{net_name}-{name}.png',I)


# -
# Hyperparams
BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 50
PATH_TO_DATASET = '/raid/users/ebarnett/OneraImages/'
TRAIN_STRIDE = int(PATCH_SIDE/2) - 1
TYPE = 3 # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands
LOAD_TRAINED = False

# -
# Datasets
train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=True, stride=TRAIN_STRIDE, transform=tr.Compose([RandomFlip(), RandomRot()]))
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

weights = torch.FloatTensor(train_dataset.weights).cuda()

test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = False, stride = TRAIN_STRIDE)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)


# -
# Model
net, net_name = FresUNet(2*13, 2), 'FresUNet'
net.cuda()

# -
# Loss
criterion = nn.NLLLoss(weight=weights) # to be used with logsoftmax output


# -
# Training
# net.load_state_dict(torch.load('net-best_epoch-1_fm-0.7394933126157746.pth.tar'))

if LOAD_TRAINED:
    net.load_state_dict(torch.load('net_final.pth.tar'))
    print('LOAD OK')
else:
    t_start = time.time()
    out_dic = train(n_epochs=50, net=net, criterion=criterion)
    t_end = time.time()
    print(out_dic)
    print('Elapsed time:')
    print(t_end - t_start)
    torch.save(net.state_dict(), 'net_final.pth.tar')
    print('SAVE OK')

# -
# Eval by plotting
t_start = time.time()
save_test_results(test_dataset)
t_end = time.time()
print('Elapsed time: {}'.format(t_end - t_start))