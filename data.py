# Imports

# PyTorch
import torch
from torch.utils.data import Dataset

# Other
import os
import random
import numpy as np

from skimage import io
from scipy.ndimage import zoom

from tqdm import tqdm as tqdm
from pandas import read_csv
from math import  ceil


# Annoying hyperparameters defined by the boy outside the functions, lazy method to incorporate

PATH_TO_DATASET = './OSCD/'
IS_PROTOTYPE = False

FP_MODIFIER = 10 # Tuning parameter, use 1 if unsure

BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 50

NORMALISE_IMGS = True

TRAIN_STRIDE = int(PATCH_SIDE/2) - 1

TYPE = 3 # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

LOAD_TRAINED = False

DATA_AUG = True

# Functions

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0] ,:s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0 ,s[0] - si[0])
    p1 = max(0 ,s[1] - si[1])

    return np.pad(I ,((0 ,p0) ,(0 ,p1)) ,'edge')


def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")

    I = np.stack((r ,g ,b) ,axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_4(path):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    I = np.stack((r ,g ,b ,nir) ,axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq20(path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif") ,2) ,s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif") ,2) ,s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif") ,2) ,s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif") ,2) ,s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif") ,2) ,s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif") ,2) ,s)

    I = np.stack((r ,g ,b ,nir ,ir1 ,ir2 ,ir3 ,nir2 ,swir2 ,swir3) ,axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq60(path):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif") ,2) ,s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif") ,2) ,s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif") ,2) ,s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif") ,2) ,s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif") ,2) ,s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif") ,2) ,s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif") ,6) ,s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif") ,6) ,s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif") ,6) ,s)

    I = np.stack((r ,g ,b ,nir ,ir1 ,ir2 ,ir3 ,nir2 ,swir2 ,swir3 ,uv ,wv ,swirc) ,axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_trio(path):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    if TYPE == 0:
        I1 = read_sentinel_img(path + '/imgs_1/')
        I2 = read_sentinel_img(path + '/imgs_2/')
    elif TYPE == 1:
        I1 = read_sentinel_img_4(path + '/imgs_1/')
        I2 = read_sentinel_img_4(path + '/imgs_2/')
    elif TYPE == 2:
        I1 = read_sentinel_img_leq20(path + '/imgs_1/')
        I2 = read_sentinel_img_leq20(path + '/imgs_2/')
    elif TYPE == 3:
        I1 = read_sentinel_img_leq60(path + '/imgs_1/')
        I2 = read_sentinel_img_leq60(path + '/imgs_2/')

    cm = io.imread(path + '/cm/cm.png', as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2 ,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0 ,0)) ,'edge')


    return I1, I2, cm

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train = True, patch_side = 96, stride = None, use_all_bands = False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train:
            fname = 'train.txt'
        else:
            fname = 'test.txt'

        #         print(path + fname)
        self.names = read_csv(path + fname).columns
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0


        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_sentinel_img_trio(self.path + im_name)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                            [self.stride*(i+1), self.stride*(j+1)])
                    self.patch_coords.append(current_patch_coords)

        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'I1': I1, 'I2': I2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}





import os
from glob import glob
import numpy as np
from tifffile import tifffile
import imageio
from PIL import Image

from torch.utils.data import Dataset
from albumentations import Compose as ACompose
from albumentations.pytorch.transforms import ToTensor as AToTensor
from albumentations.augmentations import transforms as atransforms


# Added guess for band B10
SENTINEL_BAND_STATS = {
    'mean' : np.array([1.08158484e-01, 1.21479766e-01, 1.45487537e-01, 1.58012632e-01, 1.94156398e-01, 2.59219257e-01,
                       2.83195732e-01, 2.96798923e-01, 3.01822935e-01, 3.08726458e-01, 2.08e-01, 2.37724304e-01, 1.72824851e-01])[None,None],
    'std'  : np.array([2.00349529e-01, 2.06218237e-01, 1.99808794e-01, 2.05981393e-01, 2.00533060e-01, 1.82050607e-01,
                       1.76569472e-01, 1.80955308e-01, 1.68494856e-01, 1.80597534e-01, 1.154e-01, 1.15451671e-01, 1.06993609e-01])[None,None],
}

def _sentinel_normalize(x, **kwargs):
    return (x - SENTINEL_BAND_STATS['mean']) / SENTINEL_BAND_STATS['std']

def sentinel_augmentation_valid():
    return ACompose([
        atransforms.Lambda(name='normalize', image=_sentinel_normalize),
        AToTensor(),
    ])

class OneraChngDetect(Dataset):
    def __init__(self, img_root, lab_root, normalize, bands, cities, chunk=False, patch_size=10):
        # Did not train with B10
        self.bands = bands
        self.paths_dict = {}
        for i, city in enumerate(cities):
            img_1_path = os.path.join(img_root, city, 'imgs_1_rect')
            img_2_path = os.path.join(img_root, city, 'imgs_2_rect')
            lab_path = os.path.join(lab_root, city, 'cm', f'{city}-cm.tif')
            self.paths_dict[i] = {'city': os.path.basename(city),
                                  'img_1_path': img_1_path,
                                  'img_2_path': img_2_path,
                                  'lab_path': lab_path}

        self.transform = sentinel_augmentation_valid() if normalize else AToTensor()
        self.patch_size = patch_size
        self.chunk = chunk


    def __len__(self):
        return len(self.paths_dict)

    def tiff2numpy(self, path):
        array = [tifffile.imread(os.path.join(path, band)) for band in self.bands]
        array = np.stack(array, axis=0)
        array = array.transpose(1,2,0).astype(np.float32)/10_000
        return array

    def label2numpy(self,path):
        return np.array(tifffile.imread(path)).astype(np.float32) / 255

    def _chunk(self, x, w, h, img):
        batch = []
        for row_idx in range(w):
            for col_idx in range(h):
                row_start   = self.patch_size * row_idx
                row_end     = self.patch_size * (row_idx + 1)
                col_start   = self.patch_size * col_idx
                col_end     = self.patch_size * (col_idx + 1)
                if img == True:
                    batch.append(x[:, row_start:row_end, col_start:col_end])
                else:
                    z = np.zeros((2, self.patch_size, self.patch_size))
                    k = np.where(x[row_start:row_end, col_start:col_end] == 0)
                    z[1, k[0], k[1]] = 1
                    z[0, :, :] = 1
                    z[0, k[0], k[1]] = 0
                    batch.append(z)
        return np.stack(batch)


    def __getitem__(self, idx):
        img1 = self.tiff2numpy(self.paths_dict[idx]['img_1_path'])
        img2 = self.tiff2numpy(self.paths_dict[idx]['img_2_path'])
        lab  = self.label2numpy(self.paths_dict[idx]['lab_path'])
        img1 = self.transform(image=img1)['image']
        img2 = self.transform(image=img2)['image']
        w, h = img1.shape[1] // self.patch_size, img1.shape[2] // self.patch_size
        if self.chunk:
            img1 = self._chunk(img1, w, h, True)
            img2 = self._chunk(img2, w, h, True)
            lab  = self._chunk(lab, w, h, False)

        return img1, img2, lab, w, h
