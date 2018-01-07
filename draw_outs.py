import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def draw_and_save(ct_path, mask_path, save_fpath):
    arr = np.load(ct_path)
    arr = [arr[i, :, :] for i in range(arr.shape[0])]
    arr = np.stack(arr, axis=-1)
    mask = np.load(mask_path)
    mask = mask[0,:,:,:]
    pp = PdfPages(save_fpath)
    print("arr shape {}, mask shape {}".format(arr.shape, mask.shape))
    for i in range(mask.shape[-1]):
        plt.figure()
        plt.title("ct slice {}".format(i))
        plt.imshow(arr[:,:,i])
        pp.savefig()
        plt.close()
        plt.figure()
        plt.title("mask slice {}".format(i))
        mask_slice = mask[:, :, i]
        mask_slice = mask_slice > 0.1
        plt.imshow(mask_slice)
        pp.savefig()
        plt.close()
    pp.close()

if __name__ == '__main__':
    ct_path = "/Users/anastasia/PycharmProjects/lung-segmentation-3d/Demo/test_lidc_idri/my_shape.npy"
    mask_path = "/Users/anastasia/PycharmProjects/lung-segmentation-3d/Demo/test_lidc_idri/my_shape_mask.npy"
    save_fpath = "/Users/anastasia/PycharmProjects/lung-segmentation-3d/Demo/test_lidc_idri/output_thres_01.pdf"
    draw_and_save(ct_path, mask_path, save_fpath)