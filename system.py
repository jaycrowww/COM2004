"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


def reduce_dimensions(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    
    print("shape of o.g. fvf:", feature_vectors_full.shape) # (14395,2340) - 2340 pixel features 
    
    # PCA to 40 dimensions
    covx = np.cov(feature_vectors_full, rowvar=0)
    N = covx.shape[0]
    print("N =", N)
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1)) # first 40 principal components
    v = np.fliplr(v)

    covx.shape # (2340, 2340)
    print("covx.shape:", covx.shape)
    v.shape # (2340,40)
    print("v.shape", v.shape)
    print("w.shape", w.shape)
    
    pcatrain_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v) # (14395,40) - prospective 40 features
    print("pcatrain_data:", pcatrain_data.shape)
    
    """PERFORM MULTI-DIVERGENCE to get 10 best features"""
    
    # how to reconstruct the data
    reconstructed = np.dot(pcatrain_data, v.transpose()) + np.mean(feature_vectors_full)
    print("reconstructed type:", reconstructed.shape)
    # maybe access feature selection already 
    
    #print(feature_vectors_full[:,0:10])
    #print("feature_vectors_full shape:", (feature_vectors_full[:,0:10]).shape)
    
    return feature_vectors_full[:, 0:10]


def multidivergence(class1, class2, features):
    """compute divergence between class1 and class2
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    features - the subset of features to use
    
    returns: d12 - a scalar divergence score
    """

    ndim = len(features);

    # compute mean vectors
    mu1 = np.mean(class1[:, features], axis=0)
    mu2 = np.mean(class2[:, features], axis=0)

    # compute distance between means
    dmu = mu1 - mu2

    # compute covariance and inverse covariance matrices
    cov1 = np.cov(class1[:, features], rowvar=0)
    cov2 = np.cov(class2[:, features], rowvar=0)
 
    icov1 = np.linalg.inv(cov1)
    icov2 = np.linalg.inv(cov2)

    # plug everything into the formula for multivariate gaussian divergence
    d12 = (0.5 * np.trace(np.dot(icov1, cov2) + np.dot(icov2, cov1) 
                          - 2 * np.eye(ndim)) + 0.5 * np.dot(np.dot(dmu, icov1 + icov2), dmu))

    return d12

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
    

    model_data['fvectors_train'] = fvectors_train.tolist()
    
    
    print('Feature Selection Stage')
    
    return model_data


def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    return np.repeat(labels_train[0], len(page))


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced



def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels
