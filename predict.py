import importlib
import sys
import os
from utils import savePredictions
from keras.models import load_model

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'

model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
IMAGE_SIZE = modelDefinition.IMAGE_SIZE

model_path = 'models'
model_name = model_type


model = load_model(os.path.join(model_path, model_name) + '.h5')

test_generator = test_datagen.flow_from_directory('/media/mike/Files/Data and Results/innovation-challenge-2019/Test/',
                                                  target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                  class_mode = None,
                                                  batch_size = 1,
                                                  seed = 42,
                                                  shuffle = False
                                                 )


STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
predictions = model.predict_generator(
                                      test_generator,
                                      steps = STEP_SIZE_TEST,
                                     )

savePredictions(predictions, test_generator.filenames, save_name = 'predictions/{}.csv'.format(model_name))