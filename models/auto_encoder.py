from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import numpy as np
import os

########################################################################################################################

MODEL_PATH = os.path.abspath('models') if 'models' in os.listdir() else os.getcwd()

########################################################################################################################


def auto_encoder(x_train, y_train, x_test, y_test, clearance_level, train=True, evaluate=False):
    """
    INPUT:
        x_train - numpy array of fashion_mnist encoded with clearance level for training
        y_train - numpy array of original fashion_mnist
        x_test - numpy array of fashion encoded with clearance level for testing
        y_test - numpy array of original fashion_mnist
        clearance_level [0, 1] - level to encode x_train & x_test with
        train [True, False] - train AutoEncoder model (default=True)
        evaluate [True, False] - display evaluation of AutoEncoder model (default=False)
    OUTPUT:
        decoded_train_images - x_train auto encoded with clearance level
        decoded_test_images - x_test auto encoded with clearance level

    """
    # normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = y_train.astype('float32') / 255
    y_train = y_train.reshape((len(y_train), np.prod(y_train.shape[1:])))
    y_test = y_test.astype('float32') / 255
    y_test = y_test.reshape((len(y_test), np.prod(y_test.shape[1:])))

    # if model already exists, load existing model
    if os.path.exists(MODEL_PATH + '/auto_encoder_' + str(clearance_level) + '.h5'):
        AutoEncoder = load_model(MODEL_PATH + '/auto_encoder_' + str(clearance_level) + '.h5')
        Encoder = load_model(MODEL_PATH + '/encoder_' + str(clearance_level) + '.h5')
        Decoder = load_model(MODEL_PATH + '/decoder_' + str(clearance_level) + '.h5')
    else:   # create new model
        # dimensions of "Bottle-Neck" representation
        encoding_dim = 32
        # input layer
        input_img = Input(shape=(784,))
        # hidden layers
        encoded = Dense(encoding_dim * 4, activation='relu')(input_img)
        encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
        # output layer
        decoded = Dense(784, activation='sigmoid')(decoded)

        # AutoEncoder: model that maps the input to the encoding (Encoder) to the reconstruction (Decoder)
        AutoEncoder = Model(input_img, decoded)

        # Encoder: model that maps the input to the encoding
        Encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer1 = AutoEncoder.layers[-3]
        decoder_layer2 = AutoEncoder.layers[-2]
        decoder_layer3 = AutoEncoder.layers[-1]

        # Decoder: model that maps the encoding to the reconstruction
        Decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

        # compile model for training
        AutoEncoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    if train:
        print('\n \nTraining AutoEncoder for clearance level ' + str(clearance_level))
        # train model
        AutoEncoder.fit(x_train, y_train, batch_size=256, epochs=100, verbose=1, shuffle=True,
                        validation_data=(x_test, y_test))
        # save trained models
        AutoEncoder.save(MODEL_PATH + '/auto_encoder_' + str(clearance_level) + '.h5')
        Encoder.save(MODEL_PATH + '/encoder_' + str(clearance_level) + '.h5')
        Decoder.save(MODEL_PATH + '/decoder_' + str(clearance_level) + '.h5')

    if evaluate:
        print('\tClearance Level ' + str(clearance_level) + ' Evaluation: ')
        # evaluate model on test dataset
        print(AutoEncoder.evaluate(x_test, y_test, verbose=1))

    # Encode train & test datasets
    encoded_train_images = Encoder.predict(x_train)
    encoded_test_images = Encoder.predict(x_test)

    # Decode Encoded train & test datasets
    decoded_train_images = Decoder.predict(encoded_train_images)
    decoded_test_images = Decoder.predict(encoded_test_images)

    # reshape (784 => 28x28) and re-normalize
    decoded_train_images = decoded_train_images.reshape(decoded_train_images.shape[0], 28, 28)
    decoded_train_images *= 255
    decoded_train_images = decoded_train_images.astype('uint8')
    decoded_test_images = decoded_test_images.reshape(decoded_test_images.shape[0], 28, 28)
    decoded_test_images *= 255
    decoded_test_images = decoded_test_images.astype('uint8')

    # return auto encoded x_train, x_test
    return decoded_train_images, decoded_test_images


########################################################################################################################
