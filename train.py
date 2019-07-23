from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import importlib
import sys
import os
from utils import savePredictions

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'


model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
model = modelDefinition.build_model()
IMAGE_SIZE = modelDefinition.IMAGE_SIZE

folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/Train/'
model_path = 'models'  # For saving
model_name = model_type

training_examples = 11314
validation_examples = 2831
test_examples = 999

EPOCHS = 30
BATCH_SIZE = 16


train_datagen = modelDefinition.train_datagen
test_datagen = modelDefinition.test_datagen

train_generator = train_datagen.flow_from_directory(
                                                    os.path.join(folder, 'train'),
                                                    target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical'
                                                   )

validation_generator = test_datagen.flow_from_directory(
                                                        os.path.join(folder, 'val'),
                                                        target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                        batch_size = BATCH_SIZE,
                                                        class_mode = 'categorical'
                                                       )

test_generator = test_datagen.flow_from_directory('/media/mike/Files/Data and Results/innovation-challenge-2019/Test/',
                                                  target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                  class_mode = None,
                                                  batch_size = 1,
                                                  seed = 42,
                                                  shuffle = False
                                                 )

###
logger = CSVLogger(os.path.join(model_path, model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'), exist_ok = True)  # Create folder if not present
checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, min_delta = 1e-4)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, verbose = 1, min_delta = 1e-4)
callbacks_list = [logger, checkpoint, early_stop, reduce_lr]

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
history = model.fit_generator(
                              train_generator,
                              steps_per_epoch = STEP_SIZE_TRAIN,
                              epochs = EPOCHS,
                              validation_data = validation_generator,
                              validation_steps = STEP_SIZE_VALID,
                              callbacks = callbacks_list,
                              class_weight = modelDefinition.class_weight
                             )



model.save(os.path.join(model_path, model_name) + '.h5')


#####
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
predictions = model.predict_generator(
                                      test_generator,
                                      steps = STEP_SIZE_TEST,
                                      verbose = 1
                                     )

savePredictions(predictions, test_generator.filenames, save_name = 'predictions/{}.csv'.format(model_name))