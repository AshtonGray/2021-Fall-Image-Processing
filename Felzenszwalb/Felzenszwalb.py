from skimage.segmentation import felzenszwalb as fz, mark_boundaries
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.future.graph import rag_mean_color, cut_normalized, merge_hierarchical
from skimage import color
from plot_rag_merge import _weight_mean_color, merge_mean_color
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image
    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1,3)).astype(np.float)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    # stretch matrix
    S = np.diag(1/np.sqrt(eigval))
    # compute mean of each color
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:,:,b], 100*tol), np.percentile(B[:,:,b], 100-100*tol)
            B[B<low] = low
            B[B>high] = high
        # ...rescale the color values to 0..255
        B[:,:,b] = 255 * (B[:,:,b] - B[:,:,b].min())/(B[:,:,b].max() - B[:,:,b].min())
    # return it as uint8 (byte) image
    return B.astype(np.uint8)

def segmentOriginal(original,segmented):
    segmented = np.reshape(segmented, (-1))
    labels_unique = np.unique(segmented)
    n_clusters = len(labels_unique)  # number of clusters
    print(n_clusters)
    # ----- plot the results -----
    sections = [] # contains arrays of each section and index values for each
    for i in labels_unique:
        section = np.squeeze(np.array(np.where(segmented==i)))
        sections.append(section)

    for cluster in labels_unique:
        bar = np.reshape(original.copy(), (-1, 3)) # copy the unchanged original
        for i in np.array(sections[cluster]):
            bar[i] = [255,255,255] # bar is in same shape of foo
        im2 = np.reshape(bar,original.shape)
        plt.imshow(im2)
        plt.show()

def meanShift(im):

    #im = decorrstretch(im) # Decorrelation Stretching if needed
    im = np.array(im)

    foo = np.reshape(im, (-1, 3)) # add anything else
    # Add Location
    x = np.linspace(0, 1, im.shape[1])
    y = np.linspace(0, 1, im.shape[0])
    X1, Y1 = np.meshgrid(x, y)
    X1 = X1.flatten()
    X1 = np.reshape(X1, (-1, 1))
    Y1 = Y1.flatten()
    Y1 = np.reshape(Y1, (-1, 1))
    foo = np.concatenate((foo, X1, Y1), axis=1)

    print("Shape of Meanshift Vectors Input:",foo.shape)
    # ----- Perform Meanshift -----
    bandwidth = estimate_bandwidth(foo, n_samples = 10000) # optimal bandwidth
    print("Bandwidth: ",bandwidth)
    ms = MeanShift(bandwidth=bandwidth, min_bin_freq=1000, bin_seeding = True).fit(foo)
    Y = ms.predict(foo) # array of all labels
    return Y

if __name__ == "__main__":
    # ----- Settings -----
    meanshift = False
    oversegment = False
    NCuts = False

    # Input Images
    images = ["0073MR0003970000103657E01_DRCL.tif", "0174ML0009370000105185E01_DRCL.tif",
              "0617ML0026350000301836E01_DRCL.tif", "1059ML0046560000306154E01_DRCL.tif"]
    image = plt.imread(images[0])  # File data in numpy each x,y contains array of [R,G,B]
    im = np.array(image)/255

    if oversegment:
        scale, sigma, ms = 2.5, 5, 100
        # Image 1: 10, 3.5, 17000 (17000, includes the scoop as part of the rover)
        # Image 2: 2.5, 2, 15000
        # Image 3: 2.5, 2, 15000
        # Image 4: 2.5, 5, 15000

    else:
        scale, sigma, ms = 230, 15, 30000
        # Image 1: 600, 3.5, 10000
        # Image 2: 80, 20, 17000 (heavily blur)
        # Image 3: 300, 8, 40000 ( 40,000 made sure the silver at the top of the rover was added)
        # Image 4: 230, 15, 30000 (higher blur, makes edges around object worse, however bottom left corner is an issue

    # Felzenszwalb Algorithm (returns MxN array of segment labels)
    segment_im = fz(im,scale = scale, sigma=sigma,min_size=ms) # Takes in image, scale (higher means more clusters), sigma (width of gaussian), min_size (minimum component size), multichannel (default True), channel_axis (which axis of array is channels)
    # NOTES:
    # Sigma blurs before segment
    # Oversegment sigma = 2, scale = 3, min_size = 25000
    # 1) min_size = 250000 can segment rover and background, sigma = 2

    # FZ image
    FelzColor = color.label2rgb(segment_im, im, kind='avg', bg_label=None)
    FelzColor = mark_boundaries(FelzColor, segment_im, (0, 0, 0))
    plt.imshow(FelzColor)
    plt.title("Segmented Image Using Felzenszwalb:")
    plt.show()

    # Regency Agency Graph
    if oversegment:
        if meanshift:
            merge = color.label2rgb(segment_im, im, kind='avg', bg_label=None)
            segment_im = meanShift(merge)
        else:
            rag = rag_mean_color(im,segment_im)
            if NCuts:
                labels = cut_normalized(segment_im,rag, thresh=0.2, num_cuts = 500)
                merge = color.label2rgb(labels, im, kind='avg', bg_label=None)
                merge = mark_boundaries(merge, labels, (0, 0, 0))
                # show the merged colored graph
                plt.imshow(merge)
                plt.title("Merged RAG using NCuts:")
                plt.show()
            else:
                labels = merge_hierarchical(segment_im,rag,
                                            thresh=0.5, # no 0.5 # .3 (shadow, soil, scoop in soil) no 0.2
                                            rag_copy=False,
                                            in_place_merge=True,
                                            merge_func=merge_mean_color,
                                            weight_func=_weight_mean_color)
                merge = color.label2rgb(labels, im, kind='avg', bg_label=None)
                merge = mark_boundaries(merge,labels,(0,0,0))

                # show the merged colored graph
                plt.imshow(merge)
                plt.title("Merged RAG using Hierarchical Segmentation:")
                plt.show()
            segment_im = labels

# Takes in the original image, and the labelled segmented image
segmentOriginal(im,segment_im)