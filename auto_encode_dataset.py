from models.auto_encoder import auto_encoder
from models.classifier import classifier
from encode_dataset import x_train, y_train, x_test, y_test
from encode_dataset import encoded_x_train_clr0, encoded_y_train_clr0, encoded_x_test_clr0, encoded_y_test_clr0
from encode_dataset import encoded_x_train_clr1, encoded_y_train_clr1, encoded_x_test_clr1, encoded_y_test_clr1
import numpy as np

########################################################################################################################
""" auto encode previously encoded datasets from encode_dataset.py """

# auto encode with clearance level 0
auto_encoded_x_train_clr0, auto_encoded_x_test_clr0 = auto_encoder(encoded_x_train_clr0, x_train,
                                                                   encoded_x_test_clr0, x_test, 0)
# auto encode with clearance level 1
auto_encoded_x_train_clr1, auto_encoded_x_test_clr1 = auto_encoder(encoded_x_train_clr1, x_train,
                                                                   encoded_x_test_clr1, x_test, 1)

########################################################################################################################
""" create dataset for training Classifier model in classifier.py """

# connects original (=non-encoded), encoded with 0, and encoded with 1, train datasets for training
train_images = np.concatenate([auto_encoded_x_train_clr0, auto_encoded_x_train_clr1, x_train], axis=0)
train_labels = np.concatenate([encoded_y_train_clr0, encoded_y_train_clr1, y_train])
# connects original (=non-encoded), encoded with 0, and encoded with 1, test datasets for training
test_images = np.concatenate([auto_encoded_x_test_clr0, auto_encoded_x_test_clr1, x_test], axis=0)
test_labels = np.concatenate([encoded_y_test_clr0, encoded_y_test_clr1, y_test])


def get_shuffle_configuration(array):
    """
    INPUT:
        array - numpy array
    OUTPUT:
        config - configuration to shuffle arrays of the same size
    """
    config = np.arange(array.shape[0])
    np.random.shuffle(config)
    return config


# create shuffle configuration for train and test datasets
train_config = get_shuffle_configuration(train_images)
test_config = get_shuffle_configuration(test_images)

# shuffle image and label train datasets symmetrically
train_images = train_images[train_config]
train_labels = train_labels[train_config]
# shuffle image and label test datasets symmetrically
test_images = test_images[test_config]
test_labels = test_labels[test_config]

########################################################################################################################
""" train Classifier model """

model = classifier(train_images, train_labels, test_images, test_labels)

########################################################################################################################
