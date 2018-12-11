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
    return height, width # uses maximum bounding box for each image


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

    bbox_h, bbox_w = bbox_size # one stock standard size
    nfeatures = bbox_h * bbox_w # why there are so many features for each letter_image
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        #print("what is h:", h)
        w = min(w, bbox_w) # chooses between two numbers
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures) # reshapes to be 1D array of nfeatures length/width from top left corner
        #print("len of:", len(fvectors[i,:]))
    return fvectors


def reduce_dimensions(feature_vectors_full, model):
    """ Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    #print("shape of o.g. fvf:", feature_vectors_full.shape) # (14395,2340) - 2340 pixel features 
    #reduced_noise_vector = PCA_reduce_noise(feature_vectors_full, model)
    #print("reduced_noise_vector", reduced_noise_vector.shape)

    ten_feature_vector = PCA_ten_features(feature_vectors_full, model) #(14395,10)
    #print("ten_feature_vector", ten_feature_vector.shape)
    return ten_feature_vector

"""def PCA_reduce_noise(feature_vectors_full, model):
    
    # PCA to 40 dimensions
    covx = np.cov(feature_vectors_full, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1)) # first 40 principal components
    v = np.fliplr(v)

    covx.shape # (2340, 2340)
    #print("covx.shape:", covx.shape)
    v.shape # (2340,40)
    #print("v.shape", v.shape)
    
    pcatrain_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v) # (14395,80) - prospective 40 features
    print("pcatrain_data:", pcatrain_data.shape)
    
    # Maybe future attempt to refine
    
    #!!!! ---later: PERFORM MULTI-DIVERGENCE to get 10 best features
    
    # how to reconstruct the data - to reduce noise
    reconstructed = np.dot(pcatrain_data, v.transpose()) + np.mean(feature_vectors_full)
    return reconstructed"""

def PCA_ten_features(feature_vector, model):
    # PCA to 10 dimensions
    #empty_eigenvalues = np.array()
    v = np.array(model['eigenvector'])
    
    # checking if training data eigenvalues have been calculated
    if v.size == 0:
        covx = np.cov(feature_vector, rowvar=0)
        N = covx.shape[0]
        #print("N =", N)
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1)) # first 10 principal components
        v = np.fliplr(v)

    # v must be an np.array here
    pca_processed_data = np.dot((feature_vector - np.mean(feature_vector)), v) # (14395,10) 
    model['eigenvector'] = v.tolist()
    #but V must be saved as a 
    return pca_processed_data


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
    model_data['bbox_size'] = bbox_size # standardised box size of largest possible bounding box
    model_data['eigenvector'] = np.array([]).tolist() # initialise empty array
    
    #print(bbox_size)
    #print(model_data['bbox_size'])

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
    
    print("EIGENVECTOR:""type:", type(model_data['eigenvector']))
    
    model_data['fvectors_train'] = fvectors_train.tolist()
    
    return model_data

    
def classify_page(page, model):
    """
    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    # DEBUG - print("\n\n\n!!contents of fvectors_train:", fvectors_train)
    #print("------------shape of TRAIN:", fvectors_train.shape)
    labels_train = np.array(model['labels_train'])
    # DEBUG - print("!!contents of labels_train:", labels_train)
    fvectors_test = np.array(page)
  
    # Super compact implementation of nearest neighbour 
    x = np.dot(fvectors_test, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest = np.argmax(dist, axis=1)
    
    label = labels_train[nearest]
    return label

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size'] # standard size
    images_test = utils.load_char_images(page_name)
    
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    print("###type of test reduced:", type(fvectors_test_reduced))
    return fvectors_test_reduced # np.ndarray



def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels
