"""Classification System

Altered to Skeleton code provided for assignment.

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
        
    return fvectors


def reduce_dimensions(feature_vectors_full, model):
    """ Reduces dimensions of feature vectors to 10 dimensions.
    
    Takes a list 
    

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    noise_dim = model['noise_dim']
    dim = model['dim'] 
    
    # Performs PCA to reduce to ___ dimensions for input feature vector, 
    # and reconstructs feature vector to reduce noise
    reconstructed_feature_vector = doPCA(feature_vectors_full, model, noise_dim)
    
    reduced_feature_vector = doPCA(reconstructed_feature_vector, model, dim)
    
    return reduced_feature_vector

def doPCA(feature_vector, model, d):
    """ Reducing dimensions of feature vector to 10 dimensions using PCA.
    And processes and store the eigenvectorvalues to 
    
    
    Original Code taken from 'Lab 7' includes code from sections:
        4. Computing the Principal Components
        5. Projecting the data onto the principal component axes
            
        Alterations made by Jenny Croft.
    
    """
    noise_dim = model['noise_dim']
    dim = model['dim'] 
    
    v = np.array(model['eigenvector'])
    
    # checking if training data eigenvalues have been calculated
    if v.size == 0:
        covx = np.cov(feature_vector, rowvar=0)
        N = covx.shape[0]
        #print("N =", N)
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1)) # first 10 principal components
        v = np.fliplr(v)
        # only save eigenvector for final reduction of post-noise reduction PCA 
        if d == dim:
            model['eigenvector'] = v.tolist()
    # v must be an np.array here
    pca_processed_data = np.dot((feature_vector - np.mean(feature_vector)), v) 
    if d == noise_dim:
        return np.dot(pca_processed_data, v.transpose()) + np.mean(feature_vector)

    return pca_processed_data


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
    
    # For PCA Dimension Reduction
    model_data['noise_dim'] = 50
    model_data['dim'] = 10
    
    #print(bbox_size)
    #print(model_data['bbox_size'])

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)   
    model_data['fvectors_train'] = fvectors_train.tolist()
    
    return model_data

    
def classify_page(page, model):
    """
    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    fvectors_test = np.array(page)
    
    # Perform nearest neighbour classification
    labels = nearest_neighbour(fvectors_train, labels_train, fvectors_test)
    return labels

def nearest_neighbour(fvectors_train, labels_train, fvectors_test):
    """ Performs Nearest Neighbour Classification to label test feature vectors
    
    Original Code taken from 'Lab 6' includes code from section(s):
        3. Using the classify function
            
        Alterations made by Jenny Croft. """
        
    # Super compact implementation of nearest neighbour 
    x = np.dot(fvectors_test, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest = np.argmax(dist, axis=1)
    
    return labels_train[nearest]

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
    #print("###type of test reduced:", type(fvectors_test_reduced))
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
