from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation
from keras import regularizers
from keras import optimizers

# Structure and data loading based on https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter

IMAGE_SIZE = 224

densenet = DenseNet121(include_top = False, weights = 'imagenet', input_shape = (IMAGE_SIZE,IMAGE_SIZE,3), classes = 5)

def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = 'sigmoid'))
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = optimizers.Adam(lr = 0.00005),
        metrics = ['accuracy']
    )
    
    return model


# https://keras.io/preprocessing/image/
# train_datagen = ImageDataGenerator(
#                                    rescale = 1./255,
#                                    preprocessing_function = preprocess_input,
#                                    shear_range = 0.1,
#                                    zoom_range = 0.2,
#                                    brightness_range = (-0.1, 0.1),
#                                    rotation_range = 5,
#                                    width_shift_range = 0.1,
#                                    height_shift_range = 0.1,
#                                    horizontal_flip = True,
#                                    vertical_flip = False
#                                   )

train_datagen = ImageDataGenerator(
                                   zoom_range = 0.15,  # set range for random zoom
                                   # set mode for filling points outside the input boundaries
                                   fill_mode = 'constant',
                                   cval = 0.,  # value used for fill_mode = "constant"
                                   horizontal_flip = True,  # randomly flip images
                                   vertical_flip = True,  # randomly flip images
                                  )

# class_weight = {0: 0.642,
#                 1: 1.988,
#                 2: 1.328,
#                 3: 2.503,
#                 4: 3.539}

class_weight = None