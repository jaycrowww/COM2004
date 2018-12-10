"""Train the classifier and save model.

DO NOT ALTER THIS FILE.

version: v1.0
"""
import system
import utils.utils as utils

NUM_TRAIN_PAGES = 9


def train(trainset):
    """Process training pages in directory trainset."""
    # retrieves each page and stores as list of access variables
    train_pages = ['data/' + trainset + '/page.' + str(page)
                   for page in range(1, NUM_TRAIN_PAGES+1)]
    
    # requires the method from system - process_training_data
    # sends array of page variables to system.process_training_data
    model_data = system.process_training_data(train_pages)
    utils.save_jsongz('data/model.json.gz', model_data)


if __name__ == '__main__':
    train('train')
