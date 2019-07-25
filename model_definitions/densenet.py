from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Dropout
from keras.layers import Activation, Conv2D, Concatenate, AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K


IMAGE_SIZE = 224

class DRModel():
    """
    Model definitions for DR classification
    """
    def __init__(self, base_weights = 'imagenet', base_structure = 121, base_output_loc = 'output',
                 additional_densenet_blocks = [], base_model_trainable = True,
                 pooling_type = "average", dense_layers = [], dense_regulariser = False,
                 dense_activation = 'relu', dense_dropout_rate = 0.5,
                 last_activation = 'sigmoid', loss = 'binary_crossentropy',optimizer = Adam,
                 optimizer_options = {'lr': 0.00005}, 
                 class_weight = None, zoom_range = 0.15, horizontal_flip = True, vertical_flip = True,
                 other_datagen_options = {}):
        self.base_weights = base_weights
        self.base_structure = base_structure
        self.base_output_loc = base_output_loc
        self.additional_densenet_blocks = additional_densenet_blocks
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

    def get_base_model(self, weights = 'imagenet', structure = 121, get_output_at = 'output'):
        """
        get_output_at -- 'output' or an int giving the layer to obtain as the output (eg 312 to skip conv5 or 140 to skip conv4)
        """
        if structure == 121:  # blocks = [6, 12, 24, 16]
            base_model = DenseNet121(include_top = False, weights = weights, input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))
        elif structure == 169:  # blocks = [6, 12, 32, 32]
            base_model = DenseNet169(include_top = False, weights = weights, input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))
        elif structure == 201:  # blocks = [6, 12, 48, 32]
            base_model = DenseNet201(include_top = False, weights = weights, input_shape = (IMAGE_SIZE,IMAGE_SIZE,3))
        if get_output_at == 'output':
            x = base_model.output
        else:
            x = base_model.layers[get_output_at].output
        return base_model, x

    def make_dense_layer(self, x, neurons, dense_regulariser = False, activation = 'relu', dropout_rate = 0.5):
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Dense(neurons, kernel_regularizer = l2(l = 0.001) if dense_regulariser else None)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    
    def make_dense_block(self, x, blocks):
        # Adapted from https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
        """A dense block.
        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.make_conv_block(x, 32)
        return x

    def make_conv_block(self, x, growth_rate):
        """A building block for a dense block.
        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
        # Returns
            Output tensor for the block.
        """
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias = False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias = False)(x1)
        x = Concatenate(axis = -1)([x, x1])
        return x

    def make_transition_block(self, x, reduction):
        """A transition block.
        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(K.int_shape(x)[-1] * reduction), 1, use_bias = False)(x)
        x = AveragePooling2D(2, strides = 2)(x)
        return x

    def build_model(self):
        base_model, x = self.get_base_model(self.base_weights, self.base_structure, self.base_output_loc)

        blocks = self.additional_densenet_blocks
        if blocks:
            for i in range(len(blocks) - 1):
                x = self.make_dense_block(x, blocks[i])
                x = self.make_transition_block(x, 0.5)
            x = self.make_dense_block(x, blocks[-1])

        if self.base_output_loc != "output":
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

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

        custom_model.compile(
            loss = self.loss,
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
        2: DRModel(other_datagen_options = dict(shear_range = 0.1, brightness_range = (-0.1, 0.1), rotation_range = 5, width_shift_range = 0.1, height_shift_range = 0.1)),  # previous densenet_06
        3: DRModel(dense_layers = [512, 256], dense_regulariser = True, base_model_trainable = False),  # Similar to densenet_1 structure, but without lower learning rate
        4: DRModel(class_weight = {0: 0.642, 1: 0.994, 2: 0.664, 3: 1.252, 4: 1.770}), # previous densenet_10 (weights are half of those derived from the square root of balancing)
        5: DRModel(dense_layers = [512, 256], dense_regulariser = True),
        6: DRModel(base_output_loc = 312),  # skip conv5 block
        7: DRModel(base_output_loc = 312, additional_densenet_blocks = [16], base_model_trainable = False),  # replace the last dense block with untrained
        8: DRModel(base_output_loc = 140, additional_densenet_blocks = [24, 16], base_model_trainable = False),  # replace the last two dense blocks with untrained
        9: DRModel(base_output_loc = 140, additional_densenet_blocks = [12, 6], base_model_trainable = False),
        10: DRModel(other_datagen_options = dict(brightness_range = (-0.1, 0.1), rotation_range = 3, width_shift_range = 0.1, height_shift_range = 0.1)),  # Reduced from 2
        11: DRModel(other_datagen_options = dict(brightness_range = (-0.1, 0.1), rotation_range = 3)),  # Even more reduced
        12: DRModel(base_output_loc = 140, dense_layers = [256, 128], base_model_trainable = False),
        13: DRModel(base_output_loc = 140, dense_layers = [256, 128], dense_regulariser = True, base_model_trainable = False),
        14: DRModel(base_structure = 169, base_output_loc = 140, additional_densenet_blocks = [24, 16], base_model_trainable = False),  # 8 with densenet 169
        15: DRModel(base_structure = 201, base_output_loc = 140, additional_densenet_blocks = [24, 16], base_model_trainable = False),  # 8 with densenet 201
        16: DRModel(other_datagen_options = dict(brightness_range = (-0.1, 0.1))),  # Even more reduced
        17: DRModel(other_datagen_options = dict(rotation_range = 3)),  # Even more reduced
        18: DRModel(base_structure = 169, base_output_loc = 140, additional_densenet_blocks = [12, 6], base_model_trainable = False),  # 9 with densenet 169
        19: DRModel(base_structure = 201, base_output_loc = 140, additional_densenet_blocks = [12, 6], base_model_trainable = False),  # 9 with densenet 201
        20: DRModel(other_datagen_options = dict(brightness_range = (-0.05, 0.05))),  # Reduced range
    }
    assert variant in switcher, "Model variant does not exist. Check the integer input"
    model = switcher[variant]
    return model