import os
import numpy as np
import nibabel as nib
import albumentations as A
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import config

WORK_DIR = config.WORKING_DIR

class BraTS(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, dim, num_channels, n_classes, shuffle=False, transform=None):
        super().__init__()

        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.batch_size = batch_size
        self.dim = dim
        self.num_channels = num_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)//self.batch_size
    
    def __getitem__(self, index):
        indices = np.arange(index*self.batch_size, (index+1)*self.batch_size)

        x = np.zeros((self.batch_size, *self.dim, self.num_channels), dtype=np.uint8)
        y = np.zeros((self.batch_size, *self.dim, self.n_classes), dtype=np.uint8)

        for i, j in enumerate(indices):
            image = nib.load(self.image_paths[j]).get_fdata()
            mask = nib.load(self.mask_paths[j]).get_fdata()

            x1 = zoom(image[:,:,:,0], (160/240, 192/240, 128/155))
            x2 = zoom(image[:,:,:,1], (160/240, 192/240, 128/155))
            x3 = zoom(image[:,:,:,2], (160/240, 192/240, 128/155))
            x4 = zoom(image[:,:,:,3], (160/240, 192/240, 128/155))
            image = np.stack((x1, x2, x3, x4), axis=-1)

            mask = zoom(mask, (160/240, 192/240, 128/155))
            y1 = mask == 1
            y2 = mask == 2
            y3 = mask == 3

            mask = np.stack((y1, y2, y3), axis=-1)
            
            if self.transform:
                aug = self.transform(image=image, mask=mask)
                x[i,:,:,:] = aug['image']
                y[i,:,:,:] = aug['mask']

            else:
                x[i,:,:,:] = image
                y[i,:,:,:] = mask
        
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
