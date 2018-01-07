from load_data import loadDataGeneral

import numpy as np
import pandas as pd
#import nibabel as nib
from keras.models import load_model

from scipy.misc import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure

def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    return img[:, img.shape[1] / 2, ::-1].T

img_size = 128

if __name__ == '__main__':


    seria_fpath = "/hdd1/lung-segmentation-3d/Demo/test_lidc_idri/my_shape.npy"
    save_fpath = "/hdd1/lung-segmentation-3d/Demo/test_lidc_idri/my_shape_mask.npy"
    # Load test data
    X = np.load(seria_fpath)
    X = np.expand_dims(X, axis=0)

    n_test = X.shape[0]
    inpShape = X.shape[1:]

    # Load model
    model_name = 'trained_model.hdf5' # Model should be trained with the same `append_coords`
    model = load_model(model_name)

    # Predict on test data
    pred = model.predict(X, batch_size=1)[..., 1]
    np.save(save_fpath, pred)

