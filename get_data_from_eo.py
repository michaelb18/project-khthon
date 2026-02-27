import openeo
import tempfile
import tifffile
import numpy as np
import os
import rasterio
from datetime import datetime
import math
# First, we connect to the back-end and authenticate. 
connection = openeo.connect("openeofed.dataspace.copernicus.eu")
connection.authenticate_oidc()

from pyproj import Transformer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def meters2latlon(center_lat, center_lon, height, width):
    lat_degree_m = 111111
    lon_degree_m = 111111 * math.cos(math.radians(center_lat))

    lat_margin = (height / 2) / lat_degree_m
    lon_margin = (width / 2) / lon_degree_m

    lon_min = center_lon - lon_margin
    lon_max = center_lon + lon_margin
    lat_min = center_lat - lat_margin
    lat_max = center_lat + lat_margin

    return (lon_min, lat_min, lon_max, lat_max)

def ll2px(rds, lat_list, lon_list):
    """
    rds: The DataArray returned by rioxarray
    lat_lon_bbox: A tuple of (minx, miny, maxx, maxy) in Lat/Long
    """
    """
    rds: The DataArray from rioxarray
    lat_list: List or Array of latitudes
    lon_list: List or Array of longitudes
    Returns: (pixel_cols, pixel_rows) as two numpy arrays
    """
    # 1. Setup the Transformer
    # From WGS84 (Lat/Lon) to the Image's own CRS
    img_crs = rds.rio.crs
    transformer = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)

    # 2. Convert all Lat/Lon to the Image's Coordinate System (e.g. UTM)
    target_x, target_y = transformer.transform(lon_list, lat_list)

    rows, cols = rds.rio.index(target_x, target_y)

    # Convert to standard numpy integers for plotting
    return np.array(cols), np.array(rows)

def get_image(center_lat, center_lon, height, width, days = 90):

    lon_min, lat_min, lon_max, lat_max = meters2latlon(center_lat, center_lon, height, width)

    today_date = datetime.now().date()

    month_3_data = today_date -timedelta(days=(days))
    temporal_extent = [str(month_3_data),str(today_date)]
    spatial_extent = {
        "west": lon_min, "south": lat_min,
        "east": lon_max, "north": lat_max}
    sentinel2_cube = connection.load_collection(
        "SENTINEL2_L2A", #collection chosen
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["B02", "B03", "B04","B08","B8A", "B11", "B12", "SCL"],
        max_cloud_cover=50
    )

    #sentinel2_cube = sentinel2_cube.process("mask_scl_dilation", data=sentinel2_cube, scl_band_name="SCL")
    median_image = sentinel2_cube.reduce_dimension(dimension="t", reducer="median")

    from openeo.processes import ProcessBuilder

    # define child process, use ProcessBuilder
    def scale_function(x: ProcessBuilder):
        return x * 0.0001

    # Convert to reflectance (simple multiplication)
    print("Converting to reflectance...")
    reflectance_cube= median_image.apply(scale_function)

    #def scale_function(x: ProcessBuilder):
    #    return x.linear_scale_range(0, 1, 0, 255)

    # apply scale_function to all pixels
    #visual_image= reflectance_cube.apply(scale_function)
    final_result = reflectance_cube.save_result(format="GTiff")
 
    job_title = "Field_Observation"

    job = final_result.create_job(
                    title=job_title,
                    description="This pipeline downloads and processes images of specific fields in France"
    )

    def progress_callback(status):
        # This gets called whenever the status is polled
        # We can fetch more detail directly from the job object if needed
        info = job.describe()
        progress = info.get('progress', 'Unknown %')
        print(f"🛰️ Status: {status} | Progress: {progress} | Job ID: {job.job_id}")


    try:    
        result = job.start_and_wait(
                        print=progress_callback,
                        max_poll_interval=30, 
                        connection_retry_interval=60  
                    )
        print(f"✅ Job started with ID: {job.job_id}")
    except Exception as e:
        print(f"the job failed {e}")


    with tempfile.TemporaryDirectory() as download_dir:
        download_dir = os.path.join(download_dir, 'champs3.tiff')
        image = job.download_results(download_dir)
        import rioxarray

    #    # Open the image with rioxarray
        img = rioxarray.open_rasterio(download_dir)

    #with tempfile.TemporaryDirectory() as download_dir:
    #    download_dir = os.path.join(download_dir, 'champs3.tiff')
    #    image = final_result.download(download_dir)
    #    import rioxarray

    #    img = rioxarray.open_rasterio(download_dir)

    
    return img
