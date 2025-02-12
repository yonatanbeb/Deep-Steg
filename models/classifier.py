""" create and train Classifier model """
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

########################################################################################################################

MODEL_PATH = os.path.abspath('models') if 'models' in os.listdir() else os.getcwd()
# all labels (possible network outputs) for Classifier with String description
labels = {
    # Clearance Level 1 Labels
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Boot',
    # Clearance Level 0 Labels
    10: 'Top',
    11: 'Bottom',
    12: 'Shoe',
    # No Clearance Level Label
    13: 'No Clearance'
}

########################################################################################################################
""" Classifier model training function """


def classifier(x_train, y_train, x_test, y_test, train=True, evaluate=False, num_of_classes=len(labels)):
    """
    trains Classifier model on mixed Auto Encoded dataset from auto_encode_dataset
    INPUT:
        x_train - numpy array of original, auto encoded with 0, and auto encoded with 1 fashion_mnist images
        y_train - numpy array of x_train labels
        x_test - numpy array of original, auto encoded with 0, and auto encoded with 1 fashion_mnist images
        y_test - numpy array of x_test labels
        train [True, False] - train Classifier model (default=True)
        evaluate [True, False] - display evaluation of Classifier model (default=False)
        num_of_classes - num of different labels in y_train, y_test (default=len(labels)=14)
    OUTPUT:
        Model - returns trained classifier model
    """

    # normalize all values between 0 and 1 and reshape for training
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # convert labels to categorical representation
    y_train = keras.utils.to_categorical(y_train, num_of_classes)
    y_test = keras.utils.to_categorical(y_test, num_of_classes)

    # if model already exists, load existing model
    if os.path.exists(MODEL_PATH + '/classifier.h5'):
        Model = load_model(MODEL_PATH + '/classifier.h5')
    else:   # create new model
        Model = Sequential()
        # input layer
        Model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
        # hidden layers
        Model.add(MaxPooling2D(pool_size=2))
        Model.add(Dropout(0.3))
        Model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        Model.add(MaxPooling2D(pool_size=2))
        Model.add(Dropout(0.3))
        Model.add(Flatten())
        Model.add(Dense(256, activation='relu'))
        Model.add(Dropout(0.5))
        # output layer
        Model.add(Dense(num_of_classes, activation='softmax'))

        # compile model for training
        Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if train:
        # train model
        Model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))
        # save trained model
        Model.save(MODEL_PATH + '/classifier.h5')

    if evaluate:
        # evaluate model on test dataset
        print(Model.evaluate(x_test, y_test, verbose=1))

    return Model


########################################################################################################################
