from utils import *
from os import listdir
import numpy as np
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPool2D
from keras.layers import Dense, Input, Conv2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array


def autoencoder(embedding_dim, data_path):
    input_img = []
    files = listdir(data_path)
    for f in files:
        if f is not "." and "..":
            input_img.append(img_to_array(load_img(os.path.join(data_path, "f"))))

    input_img = Input(shape=(1 * 36 * 36,))
    encoded = Dense(embedding_dim, activation='relu')(input_img)
    decoded = Dense(36 * 36, activation='relu')(encoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(input=input_img, output=encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(input_img, input_img, nb_epoch=50, batch_size=1, shuffle=True)
    embedding = encoder.predict(input_img)
    del encoder, autoencoder, input_img
    return embedding
