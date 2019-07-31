# from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, save_summary, kappa_loss, ordinal_loss
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split
from keras import backend as K

# Models: 1 = initial - 64, 256, 256, 256, 1. Batch size = 128
# 2 = batch norm at start
# 3 = divide probability features by mean
# 4 = standardise probability features
# 5 = standardise all features
# 6 = 4 + no predictions
# 7 = 3 + 128, 128, 128, 128, 1
# 8 = 7 + dropout on input

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
    # verbose = 1
else:
    # data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
    data_folder = '/home/mike/Downloads/Innovation Week 2019'
    # verbose = 2
verbose = 1
folders = ['Train/', '2015_data/Train/', '2015_data/Test/', 'aptos2019_data/Train/']
model_path = 'models'  # For saving
model_name = "multi_regress_4"
print("\n======================")
print("Training", model_name, flush = True)
print("======================\n")

# training_examples = 11314
# validation_examples = 2831
# test_examples = 999

EPOCHS = 50
BATCH_SIZE = 128
fill_type = 'mix'


### Load data
# Load x
x_columns = []
for variant in [46, 50, 51]:
    x_prob_files = ['prob_train', 'prob_2015_train', 'prob_2015_test', 'prob_aptos']
    x_probs = [pd.read_csv(os.path.join('predictions', 'densenet-{:03d}_{}.csv'.format(variant, f))) for f in x_prob_files]
    x_prob = np.vstack(x_probs)
    # x_prob /= x_prob.mean(axis = 0)
    x_prob = (x_prob - x_prob.mean(axis = 0)) / x_prob.mean(axis = 0)
    x_columns.append(x_prob)
for variant in [46, 50, 51]:
    x_pred_files = ['pred_train', 'pred_2015_train', 'pred_2015_test', 'pred_aptos']
    x_preds = [pd.read_csv(os.path.join('predictions', 'densenet-{:03d}_{}.csv'.format(variant, f)), header = None) for f in x_pred_files]
    x_pred = np.vstack(x_preds)
    x_columns.append(x_pred)

x = np.hstack(x_columns)
# x = (x - x.mean(axis = 0)) / x.mean(axis = 0)

# Load y
ys = [np.load(os.path.join(data_folder, folder + 'y_{}.npy'.format(fill_type))) for folder in folders]
y = np.vstack(ys)
classes = np.argmax(y, axis = 1)

# Observe initial accuracy
acc = np.equal(x[:, -3:], classes.reshape((-1, 1)))
print("Initial accuracy:", acc.mean(axis = 0))

# Load Test
x_test_columns = []
for variant in [46, 50, 51]:
    x_prob_files = ['prob_test']
    x_probs = [pd.read_csv(os.path.join('predictions', 'densenet-{:03d}_{}.csv'.format(variant, f))) for f in x_prob_files]
    x_prob = np.vstack(x_probs)
    x_test_columns.append(x_prob)
for variant in [46, 50, 51]:
    x_pred_files = ['pred_test']
    x_preds = [pd.read_csv(os.path.join('predictions', 'densenet-{:03d}_{}.csv'.format(variant, f)), header = None) for f in x_pred_files]
    x_pred = np.vstack(x_preds)
    x_test_columns.append(x_pred)
x_test = np.hstack(x_test_columns)



x_train, x_val, y_train, y_val = train_test_split(x, classes, test_size = 0.2, random_state = 42, stratify = classes)
# x_val, x_train, y_val, y_train = train_test_split(x, classes, test_size = 0.2, random_state = 42, stratify = classes)


# train_datagen = ImageDataGenerator(zoom_range = 0.15,  # set range for random zoom
#                                    horizontal_flip = True,
#                                    vertical_flip = True,
#                                    # set mode for filling points outside the input boundaries
#                                    fill_mode = 'constant',
#                                    cval = 0.,  # value used for fill_mode = "constant",
#                                    brightness_range = (0.9, 1.1)
#                                   )


# train_generator = train_datagen.flow(x_train, classes, batch_size = BATCH_SIZE)

def acc2(y_true, y_pred):  # Same as built in accuracy metric
    y_pred = K.round(y_pred)
    equal = K.equal(y_true, y_pred)
    return K.mean(equal)

def build_model():
    inputs = Input(shape = (18,))
    # x = BatchNormalization()(inputs)
    # x = Dropout(0.5)(inputs)
    x = Dense(64)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    for _ in range(3):
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    output = Dense(1)(x)
    model = Model(inputs = inputs, outputs = output)
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = build_model()

# class Model:
#     @staticmethod
#     def loadmodel(self, path):
#         return load_model(path, custom_objects = {'kappa_loss': kappa_loss, 'ordinal_loss': ordinal_loss})

#     def __init__(self, path):
#        self.model = self.loadmodel(path)
#        self.graph = tf.get_default_graph()

#     def predict(self, X):
#         with self.graph.as_default():
#             return self.model.predict(X)

# model_binary = Model(os.path.join(model_path, 'densenet-046_best.h5'))
# model_kappa = Model(os.path.join(model_path, 'densenet-050_best.h5'))
# model_ordinal = Model(os.path.join(model_path, 'densenet-051_best.h5'))

# model_binary = load_model(os.path.join(model_path, 'densenet-046_best.h5'))
# model_kappa = load_model(os.path.join(model_path, 'densenet-050_best.h5'), custom_objects = {'kappa_loss': kappa_loss})
# model_ordinal = load_model(os.path.join(model_path, 'densenet-051_best.h5'), custom_objects = {'ordinal_loss': ordinal_loss})

# def data_generator():
#     while True:
#         Xi, Yi = train_generator.next()
#         Xi = Xi.astype(np.uint8)
#         x_binary = model_binary.predict(Xi)
#         x_kappa = model_kappa.predict(Xi)
#         x_ordinal = model_ordinal.predict(Xi)
#         x_binary_y = (x_binary > 0.5).astype(int).sum(axis = 1) - 1
#         x_kappa_y = np.argmax(x_kappa, axis = 1)
#         x_ordinal_y = np.argmax(x_ordinal, axis = 1)

#         X = np.hstack([x_binary, x_kappa, x_ordinal, x_binary_y, x_kappa_y, x_ordinal_y])
#         yield Xi, Yi

###
checkpoint = ModelCheckpoint(os.path.join(model_path, model_name) + '_best.h5', save_best_only = True)
logger = CSVLogger(os.path.join(model_path, 'history', model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, 'history'), exist_ok = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1, min_delta = 1e-4)
callbacks_list = [logger, reduce_lr, checkpoint]

STEP_SIZE_TRAIN = x_train.shape[0] // BATCH_SIZE
# Two stage training
model.fit(
          x_train, y_train,
        #   steps_per_epoch = STEP_SIZE_TRAIN,
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          validation_data = (x_val, y_val),
          callbacks = callbacks_list,
          class_weight = None,
        #   workers = 1,
          verbose = verbose
          )


# model.save(os.path.join(model_path, model_name) + '.h5')
model = load_model(os.path.join(model_path, model_name) + '_best.h5')


#####
#y_test = model.predict(x_test, verbose = (verbose - 2) * -1 )
#y_test = np.round(np.clip(y_test, 0, 4)).astype(int).ravel()

#file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values

#os.makedirs('predictions', exist_ok = True)
#save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))
