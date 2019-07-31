from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, save_summary, kappa_loss, ordinal_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'
if len(sys.argv) > 2:
    model_variant = int(sys.argv[2])
else:
    model_variant = 1
if len(sys.argv) > 3:
    repetition = int(sys.argv[3])
else:
    repetition = None


model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
modelClass = modelDefinition.getModelVariant(model_variant)
model = modelClass.build_model()

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
    verbose = 1
    folders = ['Train/']
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
    verbose = 2
    folders = ['Train/', '2015_data/Train/', '2015_data/Test/', 'aptos2019_data/Train/']
model_path = 'models'  # For saving
model_name = "{}-{:03}{}".format( model_type, model_variant, "" if repetition is None else "_r{:02}".format(repetition) )
print("\n=====================")
print("Training", model_name, flush = True)
print("=====================\n")

# training_examples = 11314
# validation_examples = 2831
# test_examples = 999

EPOCHS = 50
BATCH_SIZE = 16
fill_type = 'mix'

### Load data
xs = [np.load(os.path.join(data_folder, folder + 'x_{}.npy'.format(fill_type)), mmap_mode = 'r') for folder in folders]
x = np.vstack(xs)
x_test = np.load(os.path.join(data_folder, 'Test/test_x_{}.npy'.format(fill_type)))

if modelClass.last_activation == "softmax":
    ys = [np.load(os.path.join(data_folder, folder + 'y_{}.npy'.format(fill_type))) for folder in folders]
    y = np.vstack(ys)
    classes = np.argmax(y, axis = 1)
else:
    ys = [np.load(os.path.join(data_folder, folder + 'y_multi_{}.npy'.format(fill_type))) for folder in folders]
    y = np.vstack(ys)
    classes = y.astype(int).sum(axis = 1) - 1

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = classes)
del x
del y

train_datagen = modelClass.get_image_datagen()

train_generator = train_datagen.flow(x_train, y_train, batch_size = BATCH_SIZE)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                 # set mode for filling points outside the input boundaries
                                 fill_mode = 'constant',
                                 cval = 0.,  # value used for fill_mode = "constant"
                                 )

val_generator = val_datagen.flow(x_val, y_val, shuffle = False, batch_size = 1)

###
best_kappa = -np.inf
val_kappas = []
best_kappa_epoch = None
STEP_SIZE_VAL = x_val.shape[0] // val_generator.batch_size
class Metrics(Callback):
    # def on_train_begin(self, logs={}):
        # self.val_kappas = []
        # self.best_kappa = -np.inf
        # self.best_kappa_epoch = None

    def on_epoch_end(self, epoch, logs={}):
        global best_kappa, best_kappa_epoch, val_kappas
        # X_val, y_val = self.validation_data[:2]
        if modelClass.last_activation == "softmax":
            _y_val = np.argmax(y_val, axis = 1)

            y_pred = self.model.predict_generator(val_generator, steps = STEP_SIZE_VAL)
            y_pred = np.argmax(y_pred, axis = 1)
        else:
            _y_val = y_val.sum(axis = 1) - 1
            
            y_pred = self.model.predict_generator(val_generator, steps = STEP_SIZE_VAL) > 0.5
            y_pred = y_pred.astype(int).sum(axis = 1) - 1

        _val_kappa = cohen_kappa_score(
            _y_val,
            y_pred, 
            weights='quadratic'
        )

        val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa > best_kappa:
            best_kappa = _val_kappa
            best_kappa_epoch = epoch
            print("Validation Kappa has improved. Saving model.")
            self.model.save(os.path.join(model_path, model_name) + '_best.h5')

        return

logger = CSVLogger(os.path.join(model_path, 'history', model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, 'history'), exist_ok = True)
# os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'), exist_ok = True)  # Create folder if not present
# checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
# early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, min_delta = 1e-4)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4)
kappa_metrics = Metrics()
callbacks_list = [logger, kappa_metrics]

STEP_SIZE_TRAIN = x_train.shape[0] // train_generator.batch_size
# Two stage training
model.fit_generator(
                    train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    epochs = EPOCHS // 2,
                    # validation_data = (x_val, y_val),
                    callbacks = callbacks_list,
                    class_weight = modelClass.class_weight,
                    workers = 4,
                    verbose = verbose
                    )

model.trainable = True

model.fit_generator(
                    train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    initial_epoch = EPOCHS // 2,
                    epochs = EPOCHS,
                    # validation_data = (x_val, y_val),
                    callbacks = callbacks_list,
                    class_weight = modelClass.class_weight,
                    workers = 4,
                    verbose = verbose
                    )


# model.save(os.path.join(model_path, model_name) + '.h5')
model = load_model(os.path.join(model_path, model_name) + '_best.h5', custom_objects = {'kappa_loss': kappa_loss, 'ordinal_loss': ordinal_loss})

#####
if modelClass.last_activation == "softmax":
    y_test = model.predict(x_test, verbose = (verbose - 2) * -1 )
    y_test = np.argmax(y_test, axis = 1)
else:
    y_test = model.predict(x_test, verbose = (verbose - 2) * -1 ) > 0.5
    y_test = y_test.astype(int).sum(axis = 1) - 1

file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values

os.makedirs('predictions', exist_ok = True)
save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))

pd.DataFrame(val_kappas).to_csv(os.path.join(model_path, 'history', model_name) + '-kappa.csv', header = False)
save_summary(model_name, best_kappa = best_kappa, epoch = best_kappa_epoch, filename = 'models/performance.csv')