import tensorflow
from tensorflow.keras.applications import EfficientNetB5
import argparse




print('[INFO] loading network...')
model = EfficientNetB5(weights='imagenet', include_top=False)

print("Showing Layer...")

#Loop over the layers in network and display them to the console
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

