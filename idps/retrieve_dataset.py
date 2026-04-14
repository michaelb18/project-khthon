import openeo
import tempfile
import tifffile
import numpy as np
import os
# First, we connect to the back-end and authenticate. 
connection = openeo.connect("openeofed.dataspace.copernicus.eu")
connection.authenticate_oidc()

from pyproj import Transformer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import datetime
import sys
sys.path.append('/home/michael/project_khthon')
from get_data_from_eo import *
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

directory = 'idp_images'
somalia = pd.read_excel('250217_cccmcluster_masterlist.xlsx')
somalia['Settlement Type'] = somalia['Settlement Type'].str.lower()
somalia = somalia[somalia['Settlement Type'] == 'planned camp/site']

for i, (lat, lon, households) in enumerate(zip(somalia['Lat'], somalia['Long'], somalia['Latest HHs'])):
    try:
        image2 = get_image(lat, lon, 5000, 5000, days = 50)
    except:
        pass
    img = (turn_into_image(image2) * 255).astype(np.uint8)
    
    Image.fromarray(img).save(os.path.join(directory, f"somalia_{i}_{households}.png"))

syria = pd.read_excel('syria_idpsites_2016earlyjan_hiu_usdos.xlsx')
for i, (lat, lon) in enumerate(zip(syria['Latitude'], syria['Longitude'])):
    try:
        image2 = get_image(lat, lon, 5000, 5000, days = 50)
    except:
        pass


    img = (turn_into_image(image2) * 255).astype(np.uint8)

    Image.fromarray(img).save(os.path.join(directory, f"syria_{i}_neg_1.png"))

haiti = pd.read_excel('hdx_sa_eq2021_r2.xlsx')

for i, (lat, lon, households) in enumerate(zip(haiti['1#1#f#2_GPS Lat'].iloc[1:], haiti['1#1#f#1_GPS Long'].iloc[1:], haiti['2#1#a#1_No# of Households'].iloc[1:])):
    try:
        image2 = get_image(lat, lon, 5000, 5000, days = 50, time = datetime(year = 2021, month = 8, day = 27).date())
    except:
        pass

    img = (turn_into_image(image2) * 255).astype(np.uint8)

    Image.fromarray(img).save(os.path.join(directory, f"haiti_{i}_{households}.png"))

sudan = pd.read_excel('./sudan_idp_camps.xlsx')
sudan.columns = sudan.iloc[0]

for i, (lat, lon, households, loc_type, loc_class, loc, locality) in enumerate(zip(sudan['_GPS_latitude'].iloc[1:], sudan['_GPS_longitude'].iloc[1:], sudan['Households'].iloc[1:, 0], sudan['Location Type'].iloc[1:], sudan['Location Classification'].iloc[1:], sudan['Location'].iloc[1:], sudan['Locality'].iloc[1:])):
    print(lat, lon)
    print(f'{loc}, {locality}')
    print(f'Location Type: {loc_type}')
    print(f'Location Class: {loc_class}')
    try:
        image2 = get_image(lat, lon, 5000, 5000, days = 50)
    except:
        pass

    img = (turn_into_image(image2) * 255).astype(np.uint8)

    Image.fromarray(img).save(os.path.join(directory, f"sudan_{i}_{households}.png"))

