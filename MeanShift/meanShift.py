from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.filters import gaussian
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import math

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

def rgb_hsi(RGBarr):
    # RGBarr is (x,3) array
    HSIarr = np.zeros(RGBarr.shape)
    '''
    R = RGBarr[:,0].flatten()/255
    G = RGBarr[:,1].flatten()/255
    B = RGBarr[:,2].flatten()/255

    print("R:", R, "G:", G, "B:", B)
    I = 1/3 * np.add(R,G,B)
    S = I
    #S = 1 - 3 * np.divide(np.minimum(R,G,B),(np.add(np.add(R,G),B) + 0.00000001))
    #S = np.where(S<0.00001,0,S)

    NUM = 0.5 * np.add(np.subtract(R,G),np.subtract(R,B))
    DEN = np.sqrt(np.add(np.square(np.subtract(R,G)),np.multiply(np.subtract(R,B),np.subtract(G,B))))
    print("Numerator", NUM, "Denominator", DEN)
    theta = np.arccos(np.divide(NUM,np.add(DEN,0.0000001)))
    H = np.where(B<=G,(theta*180/math.pi)/360,(360-(theta*180/math.pi))/360)

    HSIarr1 = np.stack([H,S,I],axis=-1)
    '''
    for j in range(RGBarr.shape[0]):
        r = RGBarr[j][0]/255
        g = RGBarr[j][1]/255
        b = RGBarr[j][2]/255
        #print("R:",r,"G:",g,"B:",b)

        i = 1 / 3 * (r + g + b)
        s = 1 - 3 * min(r, g, b) / (r + g + b + 0.00000001)
        if s < 0.00001:
            s = 0
        # calculate h
        numerator = 0.5 * ((r - g) + (r - b))
        denominator = math.sqrt(((r - g)**2) + ((r - b)*(g - b)))
        #print("Numerator",numerator,"Denominator",denominator)
        h = math.acos(numerator / (denominator + 0.00000001)) # in radians
        if b <= g:
            h = (h * 180 / math.pi)/360
        else:
            h = (360 - (h * 180 / math.pi))/360
        #print("H:",h,"S:",s,"I",i)
        HSIarr[j] = [h, s, i]

    #print("For Loop:",HSIarr,"\nElementwise:",HSIarr1)
    return HSIarr

def rgbPlot(RGBarr):
    R = RGBarr[:,0].flatten()
    G = RGBarr[:,1].flatten()
    B = RGBarr[:,2].flatten()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(R, G, B,s=0.01)
    plt.title("RGB Plot")
    plt.show()

if __name__ == "__main__":
    # ----- Settings -----
    HSI = False
    rgbplot = False
    decorrelation = False
    blur = False
    location = True

    # NOTES: HSI helps a lot and decorrelation helps a lot with both RGB and HSI
    # Expecting blur to help with any edges. Does not do so well with decorrelation
    # Just location and blurring helps a lot

    images = ["0073MR0003970000103657E01_DRCL.tif","0174ML0009370000105185E01_DRCL.tif","0617ML0026350000301836E01_DRCL.tif","1059ML0046560000306154E01_DRCL.tif"]
    image = plt.imread(images[0]) # File data in numpy each x,y contains array of [R,G,B]

    # original image PLOT THE 3d RGB ALSO THE DENSITY GRAPH
    plt.imshow(image)
    plt.title("Original Image with RGB")
    plt.show()
    im = np.array(image)

    # blur the image
    foo1 = np.reshape(im, (-1, 3))  # original

    if blur:
        im = gaussian(im, sigma=7, multichannel=True)
        plt.title("Original Image Blurred")
        plt.imshow(im)
        plt.show()
    if decorrelation:
        im = decorrstretch(im)
        plt.title("Decorrelation Stretch")
        plt.imshow(im)
        plt.show()

    foo = np.reshape(im, (-1, 3)) # add anything else
    #foo1 = foo.copy() # original

    if HSI:
        foo = rgb_hsi(foo)
        HSIimage = np.reshape(foo, im.shape)
        plt.imshow(HSIimage)
        plt.title("Original Image with HSI")
        plt.show()
    if location:
        x = np.linspace(0, 1, im.shape[1])
        y = np.linspace(0, 1, im.shape[0])
        X1, Y1 = np.meshgrid(x, y)
        X1 = X1.flatten()
        X1 = np.reshape(X1, (-1, 1))
        Y1 = Y1.flatten()
        Y1 = np.reshape(Y1, (-1, 1))
        # print(foo.shape,X1.shape,Y1.shape)
        foo = np.concatenate((foo, X1, Y1), axis=1)

    if rgbplot:
        rgbPlot(foo)

    print("Shape of Meanshift Vectors Input:",foo.shape)
    # ----- Perform Meanshift -----
    bandwidth = estimate_bandwidth(foo, n_samples = 10000) # optimal bandwidth
    print("Bandwidth: ",bandwidth)
    ms = MeanShift(bandwidth=0.2, min_bin_freq=1000, bin_seeding = True).fit(foo)
    #NOTES: Bandwidth too high (ex 500) will equal less clusters (ended up getting one cluster)
    #       Bandwidth too low (ex 10) will equal more clusters (ended up getting 13)
    #       Bin_seeding = False takes too long. Uses the kernel at all points, instead of the discretized grid of bandwidth (SET TO TRUE)
    #       min_bin_freq default = 1. Accept only bins with at least __ points as seeds (rids of holes?)
    #       Get rid of just Shadow (bandwidth = 0.2, blur True, location True) for image 0

    Y = ms.predict(foo) # array of all labels
    labels = ms.labels_
    labels_unique = np.unique(labels) # each unique cluster label
    n_clusters = len(labels_unique) # number of clusters

    print("Number of clusters: ",n_clusters)

    # ----- plot the results -----
    sections = [] # contains arrays of each section and index values for each
    for i in labels_unique:
        section = np.squeeze(np.array(np.where(Y==i)))
        sections.append(section)

    # USE TO COMBINE CLUSTERS MANUALLY
    '''
    bar = foo1.copy() # copy the unchanged original
    for cluster in [0, 1, 2, 3, 4, 6, 7]: # This array to combine the numbers of clusters
        for i in np.array(sections[cluster]):
            bar[i] = [255, 255, 255]  # bar is in same shape of foo
    im2 = np.reshape(bar, im.shape)
    plt.imshow(im2)
    plt.show()
    '''

    for cluster in labels_unique:
        bar = foo1.copy() # copy the unchanged original
        for i in np.array(sections[cluster]):
            bar[i] = [255,255,255] # bar is in same shape of foo
        im2 = np.reshape(bar,im.shape)
        plt.imshow(im2)
        plt.show()
