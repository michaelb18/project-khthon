#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/michael/project_khthon/open_street_map')

import re
import pandas as pd
from datetime import datetime
from get_osm_boxes import get_osm_features
from matplotlib import pyplot as plt


# In[2]:


def dms_to_dd(dms):
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS])\s*(\d+)°(\d+)'([\d.]+)\"([EW])"
    m = re.match(pattern, dms)
    if not m:
        return None, None
    lat = float(m[1]) + float(m[2])/60 + float(m[3])/3600
    lon = float(m[5]) + float(m[6])/60 + float(m[7])/3600
    if m[4] == 'S': lat = -lat
    if m[8] == 'W': lon = -lon
    return lat, lon


# In[3]:


df = pd.read_excel('/home/michael/project_khthon/grave_inference/AI Grave Dataset.xlsx')
print(f'Loaded {len(df)} graves')


# In[ ]:


import os
import geopandas as gpd

out_dir = '/home/michael/project_khthon/open_street_map/collected_data_5km'
os.makedirs(out_dir, exist_ok=True)

for i, row in df.iterrows():
    #if i < 47:
    #    continue
    lat, lon = dms_to_dd(row['Coordinates'])
    if lat is None:
        print(f"Skipping row {i}: could not parse '{row['Coordinates']}'")
        continue
    print('-' * 50)
    print(f"[{i+1}/{len(df)}] {row['Country']} ({lat:.4f}, {lon:.4f})")

    grave_dir = os.path.join(out_dir, f'grave_{i}')
    os.makedirs(grave_dir, exist_ok=True)

    for year in range(2009, 2027):
        dt = datetime(year, 1, 1)
        try:
            gdf = get_osm_features(lat, lon, 5000, dt=dt)
            gdf['grave_idx'] = i
            gdf['year'] = year
            gdf['lat'] = lat
            gdf['lon'] = lon

            epsg = gdf.crs.to_epsg() if gdf.crs else 4326
            fname = f'{year}_EPSG{epsg}_{lat:.4f}_{lon:.4f}.gpkg'
            path = os.path.join(grave_dir, fname)
            gdf.to_file(path, driver='GPKG')
            print(f"  {year}: {len(gdf)} features -> {fname}")
        except Exception as e:
            print(f"  {year}: ERROR - {e}")

print('\nDone!')


print(f"Data saved to {out_dir}/")
print(f"Each grave gets a subdirectory: grave_0/, grave_1/, ...")
print(f"Filenames encode year, EPSG, lat, lon: {{year}}_EPSG{{code}}_{{lat}}_{{lon}}.gpkg")
