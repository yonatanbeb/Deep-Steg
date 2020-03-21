import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import os

########################################################################################################################

labels = {
    # available output for clearance level 1
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
    # available output for clearance level 0
    10: 'Top',
    11: 'Bottom',
    12: 'Shoe',
    # available output for no clearance level
    13: 'No Clearance'
}


########################################################################################################################

def classifier(x_train, y_train, x_test, y_test, num_of_classes=len(labels)):

    # normalize all values between 0 and 1 and reshape
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

    # retrain model if exists, otherwise, create new model
    if os.path.exists(os.getcwd() + '/models/classifier.h5'):
        Model = load_model(os.getcwd() + '/models/classifier.h5')
    else:
        Model = Sequential()

        Model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
        Model.add(MaxPooling2D(pool_size=2))
        Model.add(Dropout(0.3))
        Model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        Model.add(MaxPooling2D(pool_size=2))
        Model.add(Dropout(0.3))
        Model.add(Flatten())
        Model.add(Dense(256, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(num_of_classes, activation='softmax'))

        Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    Model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))
    score = Model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test Accuracy:', score[1])
    Model.save(os.getcwd() + '/models/classifier.h5')
    return Model


def predict(image):
    Model = load_model(os.getcwd() + '/models/classifier.h5')
    if K.image_data_format() == 'channels_first':
        image = image.reshape(1, 1, 28, 28)
    else:
        image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255

    return labels[int(np.argmax(Model.predict(image)))]


########################################################################################################################
