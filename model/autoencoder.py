from keras.layers import Dense, Input, Conv2D
from keras.models import Model


def autoencoder(embedding_dim, data):
    input = Input(shape=(1 * 36 * 36,))
    encoded = Dense(embedding_dim, activation='relu')(input)
    decoded = Dense(36 * 36, activation='relu')(encoded)
    autoencoder = Model(inputs=input, outputs=decoded)
    encoder = Model(input=input, output=encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(data, data, nb_epoch=50, batch_size=1, shuffle=True)
    y = encoder.predict(data)
    del encoder, autoencoder, input, data
    return y
