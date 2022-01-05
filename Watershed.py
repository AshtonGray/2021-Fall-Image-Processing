from skimage import color
from skimage.filters import rank, gaussian
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from scipy.ndimage import label
from functools import reduce
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    # ----- Settings -----
    num = 0 # which picture
    diskRadiusGradient = 5
    diskRadiusMarkers = 25

    # ----- Image -----
    # Input Images
    images = ["0073MR0003970000103657E01_DRCL.tif", "0174ML0009370000105185E01_DRCL.tif",
              "0617ML0026350000301836E01_DRCL.tif", "1059ML0046560000306154E01_DRCL.tif"]
    image = plt.imread(images[num])  # File data in numpy each x,y contains array of [R,G,B]
    im = np.array(image)/255

    if num == 0 or num ==2:
        im = im[:,95:1500,:] # for image 1, and 2, remove border bars

    original = im.copy()
    # blur image
    #im = gaussian(im,7,multichannel=False)
    # image 1: best results, blur of 5, dRM = 25, < 35

    # convert the image to grayscale
    grayim = color.rgb2gray(im)
    #grayim = im[:,:,0] # use red channel
    grayim = rank.median(grayim, disk(5))

    plt.imshow(grayim, cmap='gray')
    plt.show()

    # use gradient on sharpest image, see what gives the sharpest image (best edges)

    # ----- Gradient -----
    # get gradient from grayscale image
    gradient = rank.gradient(grayim,disk(diskRadiusGradient))

    # ----- Markers -----
    markers = rank.gradient(grayim,disk(diskRadiusMarkers)) < 30
    markers,features = label(markers)

    print("Number of Objects found in Markers:",features)

    # ----- Watershed -----
    # perform watershed
    labels = watershed(gradient,markers)
    plt.imshow(labels)
    plt.show()

    # save for mask
    #save = color.label2rgb(labels,im,kind='avg',bg_label=0)
    #plt.imsave("mask.jpg", save)

    # Show segmented original (only for image 1)
    # original[labels != 5] = [0,0,0] # 5 = soil
    # original[labels != 44] = [0,0,0] # 44 = shadow

    plt.imshow(original)
    plt.show()

    # ----- Plot Results -----
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(grayim, cmap=plt.cm.gray)
    ax[0].set_title("Original")
    ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Local Gradient")
    ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Markers")
    ax[3].imshow(im, cmap=plt.cm.gray)
    ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
    ax[3].set_title("Segmented")
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

    # ----- Watershed / GrabCut -----

    '''bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(im.shape[:2], np.uint8)

    # newmask is the mask image I manually labelled
    newmask = cv.imread('mask1Rover.tif', 0)
    mask[newmask == 0] = 2
    mask[newmask == 255] = 1

    mask, bgdModel, fgdModel = cv.grabCut(im, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    im = im * mask[:, :, np.newaxis]
    plt.imshow(im), plt.colorbar(), plt.show()'''
