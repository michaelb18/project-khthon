import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import openeo
import tempfile, os, math
from get_data_from_eo import get_image, label_sentinel2_with_lucas, turn_into_image
from spectral_indices import NDVI, NBR, NDBI, NDWI, BSI, EVI
from read_lucas import read_lucas


connection = openeo.connect("openeofed.dataspace.copernicus.eu")


def build_lucas_pixel_dataset(
    center_lat: float,
    center_lon: float,
    height: int = 1000,
    width: int = 1000,
    days: int = 30,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    if label_columns is None:
        label_columns = ["LC1", "LC1_SPEC", "LC2"]

    img = get_image(center_lat = center_lat, center_lon = center_lon, height = height, width = width, days = days, save_path = '__tmp.tiff')
    labels = label_sentinel2_with_lucas(img, class_columns=label_columns)

    bands = np.array(img).astype(np.float32)
    bands = np.moveaxis(bands, 0, -1)  # (H, W, 12)

    indices = np.stack([
        NDVI().transform(bands),   # (H, W)
        NBR().transform(bands),    # (H, W)
        NDBI().transform(bands),   # (H, W)
        NDWI().transform(bands),   # (H, W)
        BSI().transform(bands),    # (H, W)
        EVI().transform(bands),    # (H, W)
    ], axis=-1)                    # (H, W, 6)

    mask = labels[label_columns[0]] != None
    rows, cols = np.where(mask)

    n = len(rows)
    data = {
        "row": rows,
        "col": cols,
    }

    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    for i, name in enumerate(band_names):
        data[name] = bands[rows, cols, i]

    index_names = ["NDVI", "NBR", "NDBI", "NDWI", "BSI", "EVI"]
    for i, name in enumerate(index_names):
        data[name] = indices[rows, cols, i]

    for col in label_columns:
        data[col] = labels[col][rows, cols]

    return pd.DataFrame(data)


def extract_all_pixels_from_mosaic(
    bands: list[str] | None = None,
    temporal_extent: tuple[str, str] = ("2022-01-01", "2022-12-31"),
    max_cloud_cover: int = 30,
) -> pd.DataFrame:
    if bands is None:
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

    lucas_gdf = read_lucas()
    lucas_gdf = lucas_gdf.to_crs("EPSG:4326")

    cube = (
        connection.load_collection("SENTINEL2_L2A",
            spatial_extent={"west": -10, "south": 35, "east": 40, "north": 70},
            temporal_extent=list(temporal_extent),
            bands=bands,
            max_cloud_cover=max_cloud_cover)
        .reduce_dimension(dimension="t", reducer="median")
    )

    vector_cube = cube.reduce_regions(geometries=lucas_gdf, reducer="mean")
    job = vector_cube.create_job().start_and_wait()
    result = job.get_results()
    df = result.to_pandas()

    lucas_meta = lucas_gdf[["POINT_ID", "LC1", "LC1_SPEC", "LC2", "LU1"]].copy()
    lucas_meta["POINT_ID"] = lucas_meta["POINT_ID"].astype(df["POINT_ID"].dtype)
    return df.merge(lucas_meta, on="POINT_ID", how="left")
