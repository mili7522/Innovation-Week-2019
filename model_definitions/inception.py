from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Dropout
from keras.layers import Activation, Conv2D, Concatenate, AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from utils import kappa_loss, ordinal_loss, correntropy_loss, cauchy_loss


IMAGE_SIZE = 299

class DRModel():
    """
    Model definitions for DR classification
    """
    def __init__(self, base_model_trainable = True,
                 pooling_type = "average", dense_layers = [], dense_regulariser = False,
                 dense_activation = 'relu', dense_dropout_rate = 0.5,
                 last_activation = 'sigmoid', loss = 'binary_crossentropy',optimizer = Adam,
                 optimizer_options = {'lr': 0.00005}, 
                 class_weight = None, zoom_range = 0.15, horizontal_flip = True, vertical_flip = True,
                 other_datagen_options = {}):
        self.base_model_trainable = base_model_trainable
        self.pooling_type = pooling_type
        self.dense_layers = dense_layers
        self.dense_regulariser = dense_regulariser
        self.dense_activation = dense_activation
        self.dense_dropout_rate = dense_dropout_rate
        self.last_activation = last_activation
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.class_weight = class_weight
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.other_datagen_options = other_datagen_options

    def get_base_model(self):
        base_model = InceptionResNetV2(include_top = False, weights='imagenet', input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))
        x = base_model.output
        return base_model, x

    def make_dense_layer(self, x, neurons, dense_regulariser = False, activation = 'relu', dropout_rate = 0.5):
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Dense(neurons, kernel_regularizer = l2(l = 0.001) if dense_regulariser else None)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def build_model(self):
        base_model, x = self.get_base_model()

        if self.pooling_type == "average":
            x = GlobalAveragePooling2D()(x)
        elif self.pooling_type == "max":
            x = GlobalMaxPooling2D()(x)

        for neurons in self.dense_layers:
            x = self.make_dense_layer(x, neurons, self.dense_regulariser, self.dense_activation, self.dense_dropout_rate)

        x = Dropout(0.5)(x)
        predictions = Dense(5, activation = self.last_activation)(x)
        
        custom_model = Model(inputs = base_model.input, outputs = predictions)

        base_model.trainable = self.base_model_trainable

        if self.loss == 'kappa':
            loss = kappa_loss
        elif self.loss == 'ordinal':
            loss = ordinal_loss
        elif self.loss == 'correntropy':
            loss = correntropy_loss
        elif self.loss == 'cauchy':
            loss = cauchy_loss
        else:
            loss = self.loss

        custom_model.compile(
            loss = loss,
            optimizer = self.optimizer(**self.optimizer_options),
            metrics = ['accuracy']
        )
        return custom_model


    def get_image_datagen(self):
        generator_dict = {
                          'zoom_range' : self.zoom_range,  # set range for random zoom'
                          'horizontal_flip': self.horizontal_flip,
                          'vertical_flip': self.vertical_flip,
                         }
#                                    rescale = 1./255,
#                                    preprocessing_function = preprocess_input,
#                                    shear_range = 0.1,
#                                    zoom_range = 0.2,
#                                    brightness_range = (-0.1, 0.1),
#                                    rotation_range = 5,
#                                    width_shift_range = 0.1,
#                                    height_shift_range = 0.1,
        train_datagen = ImageDataGenerator(
                                           # set mode for filling points outside the input boundaries
                                           fill_mode = 'constant',
                                           cval = 0.,  # value used for fill_mode = "constant",
                                           **generator_dict, **self.other_datagen_options
                                           )
        return train_datagen


def getModelVariant(variant):
    switcher = {
        1: DRModel(),
        2: DRModel(base_model_trainable = False, other_datagen_options = dict(brightness_range = (0.9, 1.1))),
        3: DRModel(loss = 'correntropy', base_model_trainable = False, other_datagen_options = dict(brightness_range = (0.9, 1.1))),
        4: DRModel(loss = 'cauchy', base_model_trainable = False, other_datagen_options = dict(brightness_range = (0.9, 1.1))),
    }
    assert variant in switcher, "Model variant does not exist. Check the integer input"
    model = switcher[variant]
    return model