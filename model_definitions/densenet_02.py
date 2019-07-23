from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation
from keras import regularizers
from keras import optimizers

# Balanced (square root) class weights

IMAGE_SIZE = 224
adam_optimizer_options = {'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999}

def build_model():
    base_model = DenseNet121(include_top = False, weights = 'imagenet', input_shape = (IMAGE_SIZE,IMAGE_SIZE,3), classes = 5)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    x = Dense(512, kernel_regularizer = regularizers.l2(l = 0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)
    predictions = Dense(5, activation = 'softmax')(x)

        
    custom_model = Model(inputs = base_model.input, outputs = predictions)

    base_model.trainable = False

    custom_model.compile(optimizer = optimizers.Adam(**adam_optimizer_options),
                         loss = 'categorical_crossentropy',
                         metrics = ['accuracy']
                        )

    return custom_model


# https://keras.io/preprocessing/image/
train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   preprocessing_function = preprocess_input,
                                   # shear_range = 0.2,
                                   # zoom_range = 0.2,
                                   # rotation_range = 5,
                                   # width_shift_range = 0.1,
                                   # height_shift_range = 0.1,
                                   horizontal_flip = True
                                  )

test_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function = preprocess_input)


class_weight = {0: 0.642,
                1: 1.988,
                2: 1.328,
                3: 2.503,
                4: 3.539}}