from skimage import color
from skimage.filters import rank
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ----- Settings -----
    num = 1 # (0-3)

    # Input Images
    images = ["0073MR0003970000103657E01_DRCL.tif", "0174ML0009370000105185E01_DRCL.tif",
              "0617ML0026350000301836E01_DRCL.tif", "1059ML0046560000306154E01_DRCL.tif"]

    im = plt.imread(images[num])
    if num == 0 or num == 2:
        im = im[:,95:1500,:] # for image 1 and 3, remove black lines

    mask = np.zeros(im.shape[:2],np.uint8)

    bgdModel = np.zeros((1, 65), dtype="float")
    fgdModel = np.zeros((1, 65), dtype="float")

    # ----- Get Rectangle Coordinates -----
    # image 1 = (2,2,870,800)
    # image 2 = (2,2,500,560)
    # image 3 = (525,315,1200,1400)
    # image 4 = (2,2,780,1200)
    if num == 0:
        rect = (2,2,875,805)
    elif num == 1:
        rect = (2,2,500,560)
    elif num == 2:
        rect = (525,315,1200,1400)
    elif num == 3:
        rect = (2, 2, 780, 1200)
    else:
        print("Enter value (0-3)")
        quit()


    start = rect[:2]
    end = rect[2:]
    color = (0,0,255)
    thickness = 2
    plotrect = cv.rectangle(im.copy(),start,end,color,thickness)
    plt.imshow(plotrect), plt.colorbar(), plt.show()

    cv.grabCut(im,mask,rect,bgdModel,fgdModel, iterCount=15, mode=cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    im = im * mask2[:, :, np.newaxis]
    plt.imshow(im), plt.colorbar(), plt.show()

    # ----- For even Improved Results -----
    '''
    newmask = cv.imread("Mask1.tif")
    if num == 0 or num == 2:
        newmask = newmask[:,95:1500,:] # for image 1 and 3, remove black lines
    newmask = color.rgb2gray(newmask)

    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    plt.imshow(mask), plt.colorbar(), plt.show()

    mask, bgdModel, fgdModel = cv.grabCut(im, mask, None, bgdModel, fgdModel, iterCount=10, mode=cv.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    im = im * mask[:, :, np.newaxis]
    plt.imshow(im), plt.colorbar(), plt.show()
    '''