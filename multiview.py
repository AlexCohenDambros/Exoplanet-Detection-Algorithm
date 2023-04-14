''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to combine models featuring a multiview system, or multiple representations of the problem. Product and Sum rules have been implemented.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============
import os
from pathlib import Path
import joblib

models_loaded = {}

def loading_models(models, models_names):

    """
    Description:
        Function used to load models saved on disk.

    Parameters:
        models : list
            List of saved template paths.
        models_names: list
            List with the names of the saved models.

    Return:
        None.
    """
    
    if not isinstance(models, list):
        raise ValueError("models must be a list of strings")
    
    if not isinstance(models_names, list):
        raise ValueError("models_names must be a list of strings")
    
    if len(models) != len(models_names):
        raise ValueError("models and models_names must have the same length")
    
    for model in models:
        if not isinstance(model, str):
            raise ValueError("models must be a list of strings")
    
    for model_name in models_names:
        if not isinstance(model_name, str):
            raise ValueError("models_names must be a list of strings")

    # ============= Command to load a saved model =============
    if len(models) != len(models_names):
        print("Error: Lists are not the same size!")
    else:
        for i in range(len(models)):
            models_loaded[models_names[i]] = joblib.load(models[i])

if __name__ == '__main__':

    path = Path.cwd() / "Saved_models"

    
    if os.path.exists(path):
        for subdir, _, files in os.walk(path):
            all_files_path = []
            all_file_names = []
            
            if len(files) == 0:
                continue
            
            for file in files:
                file_path = os.path.join(subdir, file)
                all_files_path.append(file_path)
                
                file_name_parts = file.rsplit(".", 1)
                file_name = file_name_parts[0]
                all_file_names.append(file_name)

            loading_models(all_files_path, all_file_names)
            
    else:
        print("Folder not found")

    print(models_loaded)