import pandas as pd
import os



def savePredictions(predictions, filenames, save_name):
    filenames = [os.path.basename(x) for x in filenames]
    df = pd.DataFrame({"Id": filenames, "Expected": predictions})
    df.to_csv(save_name, index = None)