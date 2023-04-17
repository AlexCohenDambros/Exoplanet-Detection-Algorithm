''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to load already trained and saved models.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============

import os
import joblib
from flask import abort
from pathlib import Path


# ============= Functions =============

def loading_models(model_path):

    """
    Description:
        Function used to load models saved on disk.

    Parameters:
        model_name : string
            Path to load model.

    Return:
        Return loaded model or error.
    """
    
    if not model_path or not isinstance(model_path, str):
        abort(500, "Invalid input. Model name must be a non-empty string.")

    # ============= Command to load a saved model =============
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        abort(404, "Model not found.")
        
        
def getting_models(model_name_multiview):
    
    """
    Description:
        Function used to load models in multiview.

    Parameters:
        model_name_multiview : string
            Name of the model you want to multiview.

    Return: dict
        Returns a dictionary containing the loaded models.
    """
    
    models_loaded = {}

    path = Path.cwd() / "Saved_models"

    if os.path.exists(path):
        for subdir, _, files in os.walk(path):
            if len(files) == 0 or os.path.basename(subdir) != model_name_multiview:
                continue
            
            for file in files:
                file_path = os.path.join(subdir, file)
                
                file_name_parts = file.rsplit(".", 1)
                file_name = file_name_parts[0]
                
                models_loaded[file_name] = loading_models(file_path)
                
        if models_loaded:
            return models_loaded
        else:
            abort(500, "Bad request")
            
    else:
        abort(404, "Folder not found.")