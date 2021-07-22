import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from tensor import huber_fn

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# img= np.asarray(x_train[0]).reshape(28,28)
# plt.imshow(img, cmap='binary')
# plt.show()
with strategy.scope():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Nadam()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    checkpoint = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                 save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[checkpoint])
    model.evaluate(X_test, y_test)


# model = keras.models.load_model("my_keras_model.hdf5")
print(model.predict(X_test[:3]))



