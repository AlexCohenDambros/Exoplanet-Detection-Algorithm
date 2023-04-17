''' Author: Alex Cohen Dambrós Lopes

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Code used to combine models featuring a multiview system, or multiple representations of the problem. Product and Sum rules have been implemented.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

'''

# ============= Imports =============

import pandas as pd
from General_Functions import load_saved_models


# ============= Functions =============

def multiview(model, mode, X_test_local, X_test_global):
    
    # Checks if model is a string and not empty
    if not isinstance(model, str):
        raise TypeError('The model parameter must be a string')
    elif not model.strip():
        raise ValueError('The model parameter cannot be an empty string')
    
    # Check if mode is one of two valid options
    if mode not in ['Produto', 'Soma']:
        raise ValueError('The mode parameter must be "Produto" or "Soma"')
    
    # Check if X_test_local and X_test_global is not empty
    if X_test_local is not None:
        raise ValueError('X_test_local parameter cannot be empty')
    if X_test_global is not None:
        raise ValueError('X_test_global parameter cannot be empty')
    
    dict_multiview = load_saved_models.getting_models(model)
    
    # Separating the values ​​of each key into different variables
    model_local = dict_multiview[[ch for ch in dict_multiview.keys() if ch.endswith("_local")][0]]
    model_global = dict_multiview[[ch for ch in dict_multiview.keys() if ch.endswith("_global")][0]]
    
    local_prob = model_local.predict_proba(X_test_local)[:,1]
    global_prob = model_global.predict_proba(X_test_global)[:,1]
    
    if mode == "Produto":
        product_prob = local_prob * global_prob
        predicted_proba_norm = product_prob / (local_prob + global_prob) # Normalize the results
    else:
        sum_prob = local_prob + global_prob
        predicted_proba_norm = sum_prob / (local_prob + global_prob) # Normalize the results
    
    if len(X_test_local) == predicted_proba_norm.shape[0] and len(X_test_global) == predicted_proba_norm.shape[0]:
        return pd.DataFrame(data = {"Predicted Candidates": predicted_proba_norm})