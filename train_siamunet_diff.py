import os
from tqdm import tqdm
import argparse
import random
import numpy as np

import torch
from models.siamunet_diff import SiamUnet_diff
from data import OneraChngDetect

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, precision_score, recall_score


def calc_loss(pred, target, metrics, bce_weight=0.5):
    loss = F.binary_cross_entropy_with_logits(pred, target)
    f1 = f1_score(np.argmax(target.data.cpu().numpy(), axis=1).flatten().reshape(-1,1),
                               np.argmax(pred.data.cpu().numpy(), axis=1).flatten().reshape(-1,1), average='weighted')
    return loss, f1

def train_model(model, optimizer, dataloaders, base_dir, num_epochs=25):
    best_loss = 1e10
    save_metrics = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        epoch_metrics = {}
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            phase_metrics = {}
            epoch_samples = 0
            for idx, (img1, img2, labs, w, h) in enumerate(tqdm(dataloaders[phase], total = len(dataloaders[phase]))):
                img1 = img1.squeeze(0).cuda()
                img2 = img2.squeeze(0).cuda()
                labs = labs.squeeze(0).cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img1, img2)
                    loss, f1 = calc_loss(outputs, labs, phase_metrics)
                    print(loss, f1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_samples += img1.size(0)
            for k in phase_metrics.keys():
                epoch_metrics[f'{k}_{phase}'] = phase_metrics[k] / epoch_samples

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='HorizontalFlip, RandomRotate90', type=str)
parser.add_argument('--n_class', default=10, type=int)
parser.add_argument('--exp_logs', default='/home/ebarnett/moco/chng_dtct/test/')
args = parser.parse_args()
_ = random.seed(3)
_ = torch.manual_seed(3)
_ = np.random.seed(3)
os.makedirs(os.path.join(args.exp_logs, 'imgs'), exist_ok=True)

bands           = ['B01.tif','B02.tif','B03.tif','B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif',
                   'B09.tif', 'B10.tif','B11.tif', 'B12.tif']
train_cities    = ['aguasclaras', 'bercy', 'bordeaux', 'nantes', 'paris', 'rennes', 'saclay_e', 'abudhabi',
                     'cupertino', 'pisa', 'beihai', 'hongkong', 'beirut', 'mumbai']
test_cities     = ['brasilia', 'montpellier', 'norcia', 'rio', 'saclay_w', 'valencia', 'dubai', 'lasvegas', 'milano',
                   'chongqing']



train_set = OneraChngDetect(img_root = '/raid/users/ebarnett/Onera/',
                            lab_root = '/raid/users/ebarnett/Onera/labels/',
                            cities = train_cities,
                            bands = bands,
                            normalize=True,
                            chunk = True,
                            patch_size = 16)


test_set  = OneraChngDetect(img_root = '/raid/users/ebarnett/Onera/',
                            lab_root = '/raid/users/ebarnett/Onera/labels/',
                            cities = test_cities,
                            bands = bands,
                            normalize=True,
                            chunk = True,
                            patch_size = 16)


model = SiamUnet_diff(input_nbr = 13, label_nbr = 2)
model.cuda()
train_loader   = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

test_loader    = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
dataloaders    = {'train': train_loader, 'test': test_loader}

# Get Optimizer / Scheduler
optimizer_ft        = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
train_model(model, optimizer_ft, dataloaders=dataloaders, num_epochs=50, base_dir=args.exp_logs)
