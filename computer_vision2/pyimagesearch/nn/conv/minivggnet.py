from tensorflow.keras import layers
from keras import backend as K
from tensorflow.keras.models import Model


class miniVGG:
    def __init__(self,height, width, depth,classes, include_top=True, pooling='avg'):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes
        self.include_top = include_top
        self.pooling = pooling

    def build(self):
        input_shape = (self.height, self.width, self.depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (self.height, self.width, self.depth)
            chanDim = 1
        input = layers.Input(shape=input_shape, name='input')
        # Block 1
        x = layers.Conv2D(
            32, (3, 3), activation='swish', padding='same', name='block1_conv1')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            32, (3, 3), activation='swish', padding='same', name='block1_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(
            64, (3, 3), activation='swish', padding='same', name='block2_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            64, (3, 3), activation='swish', padding='same', name='block2_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15, name='drop_1')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name='drop_2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(
            512, (3, 3), activation='swish', padding='same', name='block5_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            512, (3, 3), activation='swish', padding='same', name='block5_conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            512 , (3, 3), activation='swish', padding='same', name='block5_conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25, name='drop_3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if self.include_top:
            # Classification block
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(512, activation='swish', name='fc1')(x)
            x = layers.Dense(512, activation='swish', name='fc2')(x)
            x = layers.Dropout(0.3, name='drop_4')(x)
            x = layers.Dense(512, activation='swish', name='fc3')(x)
            x = layers.Dense(512, activation='swish', name='fc4')(x)
            x = layers.Dropout(0.35, name='drop_5')(x)

            x = layers.Dense(self.classes, activation='softmax',
                             name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)


        model = Model(inputs=input, outputs=x, name='miniVGG' )

        return model

if __name__ == '__main__' :
    print('ok')