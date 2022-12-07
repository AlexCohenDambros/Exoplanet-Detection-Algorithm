# coding = utf-8
# Author: Alex Cohen Dambrós Lopes 

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Getting all light curves from space telescopes and running their pre-processing
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This is an example code used to obtain light curves from space telescopes and preprocess 
them automatically

"""


# ============= Imports =============
import pandas as pd
import numpy as np
import time
import lightkurve as lk
import warnings


# ============= RandomState =============
random_state = np.random.RandomState(123)


# ============= Warnings =============
warnings.simplefilter("ignore")