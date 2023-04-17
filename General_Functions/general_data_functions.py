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

import os
import time
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# ============= Warnings =============

warnings.simplefilter("ignore")


# ============= Read Datasets =============

def read_dataset(telescope_name=None):

    telescopes = ['TESS', 'K2', 'KEPLER']

    if telescope_name is None:
        raise ValueError("read_dataset() missing 1 required positional argument: 'telescope_name'")
        
    if telescope_name.upper() not in telescopes:
        raise ValueError(f"Invalid telescope name. It must be one of the following: {', '.join(telescopes)}")

    telescope_path = f'Datasets/{telescope_name.upper()}/*.csv'
    telescope_csv_paths = glob.glob(telescope_path)

    if not telescope_csv_paths:
        raise ValueError(f"No CSV files found for telescope {telescope_name}")

    csv_data = ''
    with open(telescope_csv_paths[0], 'r') as f:
        csv_data = '\n'.join(
            list(filter(lambda a: not a.startswith('#'), f.readlines())))[1:]

    dataset = pd.read_csv(StringIO(csv_data), on_bad_lines='skip')

    if telescope_name.upper() == 'TESS':
        names_disp = {"FP": 'FALSE POSITIVE', "PC": 'CANDIDATE',
                      "CP": 'CONFIRMED', "FA": 'FALSE POSITIVE', "KP": 'CONFIRMED'}
        dataset.replace({'tfopwg_disp': names_disp}, inplace=True)

    return dataset


def download_all_datasets():
    """
        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Bot created using web spring to automatically download data from NASA
        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        This algorithm automatically selects and downloads data from the K2, KEPLER, and TESS space 
        telescopes from the NASA Exoplanet Archive website.
    """
    
    # ============= Create folders =============
    create_datasets()
    
    # ============= get path to download =============
    get_current_path = os.getcwd() + "\\Datasets"

    # ============= URL =============
    urlTESS = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI'
    urlKEPLER = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative'
    urlK2 = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc'

    list_urls = {"TESS": urlTESS, "KEPLER": urlKEPLER, "K2": urlK2 }
    
    # ============= Passing through all URLs =============
    for telescope, url in list_urls.items():

        try:
            
            # ============= Options =============
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            
            old_latest_file = get_current_path
            
            # ============= Directory =============
            prefs = {"profile.default_content_settings.popups": 0,
                    "download.default_directory": old_latest_file + f"\\{telescope}\\",
                    "directory_upgrade": True}
            chrome_options.add_experimental_option("prefs", prefs)

            # ============= Path =============
            PATH = "/path/to/chromedriver"
            driver = webdriver.Chrome(PATH, chrome_options=chrome_options)
            
            # ============= Open URL =============
            driver.get(url)
            
            time.sleep(5)
    
            driver.execute_script(
                """
                
                ;(async () => {
                [...document.querySelectorAll(`div`)]
                .filter(a => a.textContent === `Download All Columns`)
                .filter(a => a.className.includes(`sub_item_text`))[0]
                .parentElement.click() 
                await new Promise(r => setTimeout(r, 1000));
                [...document.querySelectorAll(`div`)]
                .filter(a => a.textContent === `Download All Rows`)
                .filter(a => a.className.includes(`sub_item_text`))[0]
                .parentElement.click() 
                await new Promise(r => setTimeout(r, 1000));
                [...document.querySelectorAll(`div`)]
                .filter(a => a.textContent === `Download Table`)
                .filter(a => a.className.includes(`sub_item_text`))[0]
                .parentElement.click()   
                })();
                
                """)    
            
            time.sleep(3)

            # Wait for the download to complete to continue
            t = 0
            while(t < 1000):
                time.sleep(3)
                list_of_files = glob.glob(get_current_path + f"\\{telescope}\\*.csv")
                new_latest_file = max(list_of_files, key=os.path.getctime) # Pega o ultimo arquivo baixado
                
                if os.path.isfile(new_latest_file):
                    break

                t += 1

            time.sleep(15)
            driver.close()
            driver.quit()
            
            print(f"Downloaded data from the telescope: {telescope}")

        except Exception as error:
            print(error)
        
    
def create_datasets():
    
    get_current_path = os.getcwd()

    # Path
    path = os.path.join(get_current_path, 'Datasets')
    
    # ============= Create the directory =============
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok = True)
        else:
            os.makedirs(path, exist_ok = True)
        
    except OSError as error:
        print("Directory can not be created: ", error)
    
    
    # ============= Create subfolder =============
    
    # TESS
    sub_path = path+'\\TESS'
    os.makedirs(sub_path, exist_ok = True)
    # KEPLER
    sub_path = path+'\\KEPLER'
    os.makedirs(sub_path, exist_ok = True)
    # K2
    sub_path = path+'\\K2'
    os.makedirs(sub_path, exist_ok = True)
    
    print("Directory created successfully")