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
import matplotlib.pyplot as plt
import scipy.linalg


# KEY - must reduce dimensions with this function
def reduce_dimensions(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    
    # Principle Component Analysis 
    covx = np.cov(feature_vectors_full, rowvar=0)
    N = covx.shape[0]
    w, v = np.linalg.eigh(covx, eigvals=(N - 10, N - 1)) # first 40 principal components
    v = np.fliplr(v)
    print("v:", v.shape)
    
    pca_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
    #reconstructed = np.dot(pcatrain_data, v.transpose()) + np.mean(feature_vectors_full)
    
    # Feature Selection
    

    return pca_data


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
    
    # initialises empty np array for potential feature vectors of height of len(images), and width nfeatures
    fvectors = np.empty((len(images), nfeatures))
    
    # i will be a number starting at 1 - enumerate creates an automatic counter
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255 # makes padded_images of white pixels - uses broadcasting
        #print("padded_image: ", i, ":")
        #print(padded_image)
        h, w = image.shape
        
        # image bounding boxes all vary
        # print("image shape:", image.shape)
        
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w] # tranposing image across to np array from image
        fvectors[i, :] = padded_image.reshape(1, nfeatures) # stores each transposed image as a np.1D.array in a list of vectors.
        #print("fvectors[", i, ",:]=", fvectors[i,:])
        #print("shape of fvectors:", fvectors[i,:].shape)
    #print("nfvectors:", len(fvectors), "fvectors:", fvectors)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    
    
    # Lecturer said that 'reading data' does not need to be modified
    # if you decide to let the letter be placed in the box from the left.
    # whereas it would need to be rewritten if you wishes to stretched the letter

    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)
    
    # Testing Reading Data
    print("**** print:", labels_train)
    print("shape of labels_train:", labels_train.shape) #14395 labels read in as np.array 
    print("length of images_train:", len(images_train)) # 14395 images read in as list 
    
    
    # Extracts all features from training data - images --> featurevectors
    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    # list of np.array 
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
    
    # Testing Extracting Features
    print("--- fvectors_train_full", fvectors_train_full, "shape:", fvectors_train_full.shape) # 2D Np array, shape - 14935, 2340
    print("++ fvectors_train_full[0]:", fvectors_train_full[0], "image dimensions:", fvectors_train_full[0].shape)
    print("++ fvectors_train_full[1]:", fvectors_train_full[1], "image dimensions:", fvectors_train_full[1].shape)
    

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
# to be improved - reduce_dimensions    
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


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
    # Perform the dimensionality reduction with test data.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model) # NOTE: haven't used model yet in reduce_dimensions
    return fvectors_test_reduced

# part of Evaluate.py?
def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    
    # retrieves processed fVectors (1D np.array of pixel colours)
    fvectors_train = np.array(model['fvectors_train'])
    
    # retrieves corresponding labels for each fVector
    labels_train = np.array(model['labels_train'])
    
    fvectors_test = np.array(page)
    
    labels_test = np.array(model['labels_test'])
    
    #labels_
    
    # Nearest Neighbour Classification 
    """ DESIGN """
    """Perform nearest neighbour classification."""

    # Use all feature is no feature parameter has been supplied
    
    features = None; 
    if features is None:
        features=np.arange(0, fvectors_train.shape[1])

    # Select the desired features from the training and test data
    train = fvectors_train[:, features]
    test = fvectors_test[:, features]
    
    # Super compact implementation of nearest neighbour 
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1)
    mdist=np.max(dist, axis=1)
    label = labels_train[0, nearest]
    score = (100.0 * sum(labels_test[0, :] == label)) / label.shape[0]

    # Construct a confusion matrix
    nclasses = np.max(np.hstack((labels_test, labels_train)))
    confusions=np.zeros((nclasses, nclasses))
    for i in range(labels_test.shape[1]):
        confusions[labels_test[0, i] - 1, label[i] - 1] += 1

    return score, confusions
    
    
    
    # returns first label of train and repeats it in a 1D array of len(page length)
    #return np.repeat(labels_train[0], len(page))



# EXTENSION - only attempt after preliminary system is running
    # possibility to use cosine similarity with dictionary - tf measure
    # only check words not with Capital letter.
def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels
