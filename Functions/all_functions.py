# coding = utf-8
# Author: Alex Cohen Dambrós Lopes 

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
All functions created and used in the project
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Links to get the data: 

KEPLER: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
TESS: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
K2: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc

"""


# ============= Imports =============
import pandas as pd
import numpy as np
import lightkurve as lk
from selenium import webdriver
from datetime import datetime
from selenium.webdriver.chrome.options import Options
import warnings
import time
import glob


# ============= RandomState =============
random_state = np.random.RandomState(123)


# ============= Warnings =============
warnings.simplefilter("ignore")


# ============= Read Datasets =============
def read_dataset(telescope_name = None, header_line = 0):
    
    telescopes = ['tess', 'k2', 'kepler']
    
    if telescope_name == None: 
        raise TypeError("read_dataset() missing 1 required positional argument: 'telescope_name'")
    
    elif telescope_name in telescopes:
        
        if telescope_name == 'tess': 
            telescope_path = 'Datasets\TESS'
            
        elif telescope_name == 'k2':
            telescope_path = 'Datasets\K2'
        
        elif telescope_name == 'kepler':
            telescope_path = 'Datasets\KEPLER'
            
    else:
        raise ValueError(f"The argument telescope_name does not have the telescope data: '{telescope_name}'")
    
    telescope_path = glob.glob(telescope_path+'\*.csv')
    dataset = pd.read_csv(telescope_path[0], header = header_line, on_bad_lines='skip')
    

    if telescope_name == 'tess':
        names_disp = {"FP" : 'FALSE POSITIVE', "PC" : 'CANDIDATE', "CP": 'CONFIRMED', "FA": 'FALSE POSITIVE', "KP": 'CONFIRMED'}
        dataset.replace({'tfopwg_disp': names_disp}, inplace=True)

    return dataset


def download_all_data():

    """
        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Bot created using web spring to automatically download data from NASA
        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        
        This algorithm automatically selects and downloads data from the K2, KEPLER, and TESS space 
        telescopes from the NASA Exoplanet Archive website.
    """
    
    
    # ============= Options =============
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)


    # ============= Path =============
    PATH = "/path/to/chromedriver"
    driver = webdriver.Chrome(PATH) # chrome_options = chrome_options  


    # ============= URL =============
    urlTESS = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI'
    urlKEPLER = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative'
    urlK2 = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc'

    list_urls = [urlTESS, urlKEPLER, urlK2]


    # ============= Passing through all URLs ============= 
    for url in list_urls:
    
        try:
            # ============= Open URL =============
            driver.get(url)

        except Exception as error:
            print(error)
