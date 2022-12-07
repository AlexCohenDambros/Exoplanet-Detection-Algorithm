# coding = utf-8
# Author: Alex Cohen Dambrós Lopes 

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Bot created using web spring to automatically download data from NASA
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

This algorithm automatically selects and downloads data from the K2, KEPLER, and TESS space 
telescopes from the NASA Exoplanet Archive website.

Links to get the data: 

KEPLER: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
TESS: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
K2: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc

"""


# ============= Imports =============
import time
from selenium import webdriver
from datetime import datetime
from selenium.webdriver.chrome.options import Options


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


# ============= passing through all URLs ============= 

for url in list_urls:
    
    try:
        # ============= Open URL =============
        driver.get(url)

    except Exception as error:
        print(error)
    
    
