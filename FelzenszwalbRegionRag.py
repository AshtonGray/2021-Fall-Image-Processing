from skimage.segmentation import felzenszwalb as fz, mark_boundaries
from skimage.feature import canny
from skimage.filters import sobel, difference_of_gaussians, gaussian
from skimage.future.graph import merge_hierarchical, rag_boundary, show_rag
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

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
def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }
def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

if __name__ == "__main__":
    # ----- Settings -----
    sobelEdge = False
    blur = False # only applies to sobel
    cannyEdge = True
    diffGaussEdge = False # 697

    # Input Images
    images = ["0073MR0003970000103657E01_DRCL.tif", "0174ML0009370000105185E01_DRCL.tif",
              "0617ML0026350000301836E01_DRCL.tif", "1059ML0046560000306154E01_DRCL.tif"]
    image = plt.imread(images[1])  # File data in numpy each x,y contains array of [R,G,B]
    im = np.array(image)/255

    # oversegmentation
    scale, sigma, ms = 10, 2, 100

    # Felzenszwalb Algorithm (returns MxN array of segment labels)
    segment_im = fz(im,scale = scale, sigma=sigma,min_size=ms) # Takes in image, scale (higher means more clusters), sigma (width of gaussian), min_size (minimum component size), multichannel (default True), channel_axis (which axis of array is channels)

    # FZ image
    FelzColor = color.label2rgb(segment_im, im, kind='avg', bg_label=None)
    FelzColor = mark_boundaries(FelzColor, segment_im, (0, 0, 0))
    plt.imshow(FelzColor)
    plt.title("Segmented Image Using Felzenszwalb:")
    plt.show()

    # Construct Edge Map
    if sobelEdge:
        if blur:
            edges = sobel(color.rgb2gray(gaussian(im, sigma = 5)))
        else:
            edges = sobel(color.rgb2gray(im))

    elif cannyEdge:
        edges = canny(color.rgb2gray(im), sigma = 3, low_threshold=0.94,high_threshold=0.995, use_quantiles=True)
        edges = np.asarray(edges, dtype=np.float32)

    elif diffGaussEdge:
        edges = difference_of_gaussians(color.rgb2gray(im), 5, 9)

    else:
        print("You Choose at Least One Edge")
        quit()


    # show edge graph
    plt.imshow(edges)
    plt.title("Edges Image")
    plt.show()

    # Create the boundary RAG
    rag = rag_boundary(segment_im, edges)
    show_rag(segment_im,rag,im)
    plt.title("Initial RAG")
    plt.show()

    # Hierarchical Merging
    labels = merge_hierarchical(segment_im,rag,
                               thresh=0.1,
                               rag_copy=False,
                               in_place_merge=True,
                               merge_func=merge_boundary,
                               weight_func=weight_boundary)

    show_rag(segment_im, rag, im)
    plt.title("RAG after Hierarchical Merge")
    plt.show()

    merge = color.label2rgb(labels, im, kind='avg', bg_label=None)
    #merge = mark_boundaries(merge,labels,(0,0,0))

    # show the merged colored graph
    plt.imshow(merge)
    plt.title("Merged RAG using Hierarchical Segmentation:")
    plt.show()
    segment_im = labels

    # Takes in the original image, and the labelled segmented image
    segmentOriginal(im,segment_im)