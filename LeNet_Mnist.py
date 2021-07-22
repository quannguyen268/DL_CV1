from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
import keras
class LeNet:
    def build(width,height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
        # first block: CONV=>RELU=>POOLING
        model.add(Conv2D(20, (5,5), strides=(1,1),padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # second block: CONV =< RELU => POOLING
        model.add(Conv2D(50, (5,5),strides=(1,1), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # FC block
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()

        return model



from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] accessing MNIST...")
X, Y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X=X.astype("float32")/255.0
Y = Y.astype("int")
if K.image_data_format() == "channels_first":
    X = X.reshape(X.shape[0], 1, 28, 28)

else:
    X = X.reshape(X.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25, random_state=42)
print(trainY[:5])
# convert labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

print("[INFO] compiling model...")

opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint = keras.callbacks.ModelCheckpoint("LeNet_MNIST.h5", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=128, epochs=20, verbose=1, callbacks=[checkpoint, early_stopping])
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), history.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()




