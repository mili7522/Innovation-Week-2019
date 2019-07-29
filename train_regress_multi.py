from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, save_summary, kappa_loss, ordinal_loss
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout


if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
    verbose = 1
    folders = ['Train/']
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
    verbose = 2
    folders = ['Train/', '2015_data/Train/', '2015_data/Test/', 'aptos2019_data/Train/']
model_path = 'models'  # For saving
model_name = "multi_regress"
print("\n======================")
print("Training", model_name, flush = True)
print("======================\n")

# training_examples = 11314
# validation_examples = 2831
# test_examples = 999

EPOCHS = 50
BATCH_SIZE = 16
fill_type = 'mix'

### Load data
xs = [np.load(os.path.join(data_folder, folder + 'x_{}.npy'.format(fill_type)), mmap_mode = 'r') for folder in folders]
x_train = np.vstack(xs)
x_test = np.load(os.path.join(data_folder, 'Test/test_x_{}.npy'.format(fill_type)))


ys = [np.load(os.path.join(data_folder, folder + 'y_{}.npy'.format(fill_type))) for folder in folders]
y = np.vstack(ys)
classes = np.argmax(y, axis = 1)

ys = [np.load(os.path.join(data_folder, folder + 'y_multi_{}.npy'.format(fill_type))) for folder in folders]
y_multi = np.vstack(ys)


train_datagen = ImageDataGenerator(zoom_range = 0.15,  # set range for random zoom
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   # set mode for filling points outside the input boundaries
                                   fill_mode = 'constant',
                                   cval = 0.,  # value used for fill_mode = "constant",
                                   brightness_range = (0.9, 1.1)
                                  )


train_generator = train_datagen.flow(x_train, classes, batch_size = BATCH_SIZE)

model_binary = load_model(os.path.join(model_path, 'densenet-046_best.h5'))
model_kappa = load_model(os.path.join(model_path, 'densenet-050_best.h5'), custom_objects = {'kappa_loss': kappa_loss})
model_ordinal = load_model(os.path.join(model_path, 'densenet-051_best.h5'), custom_objects = {'ordinal_loss': ordinal_loss})

def data_generator():
    while True:
        Xi, Yi = train_generator.next()
        Xi = Xi.astype(np.uint8)
        x_binary = model_binary.predict(Xi)
        x_kappa = model_kappa.predict(Xi)
        x_ordinal = model_ordinal.predict(Xi)

        X = np.hstack([x_binary, x_kappa, x_ordinal])
        yield X, Yi
        

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim = 15))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    for i in range(3):
        model.add(Dense(256, input_dim = 15))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = build_model()

###
logger = CSVLogger(os.path.join(model_path, 'history', model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, 'history'), exist_ok = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4)
callbacks_list = [logger, reduce_lr]

STEP_SIZE_TRAIN = x_train.shape[0] // BATCH_SIZE
# Two stage training
model.fit_generator(
                    data_generator(),
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    epochs = EPOCHS,
                    # validation_data = (x_val, y_val),
                    callbacks = callbacks_list,
                    class_weight = None,
                    workers = 1,
                    verbose = verbose
                    )


model.save(os.path.join(model_path, model_name) + '.h5')

#####
y_test = model.predict(x_test, verbose = (verbose - 2) * -1 )
y_test = np.round(y_test).astype(int)

file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values

os.makedirs('predictions', exist_ok = True)
save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))
