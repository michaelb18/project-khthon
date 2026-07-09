import osmnx as ox
import numpy as np
import geopandas as gpd
from datetime import datetime

ox.settings.overpass_endpoint = "https://overpass-api.de/api"


def get_osm_features(lat: float, lon: float, dist: float, dt: datetime | None = None) -> gpd.GeoDataFrame:
    if dt is not None:
        date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ox.settings.overpass_settings = f"[out:json][date:'{date_str}']{{maxsize}}"
    else:
        ox.settings.overpass_settings = "[out:json]{maxsize}"
    lulc_tags = {
        "landuse": True,
        "amenity": True,
        "building": True,
        "natural": True,
    }

    bbox = ox.utils_geo.bbox_from_point(point=(lat, lon), dist=dist)
    gdf = ox.features_from_bbox(bbox=bbox, tags=lulc_tags)

    gdf['building'] = gdf['building'].replace('yes', np.nan)
    target_cols = ['landuse', 'natural', 'building', 'amenity',
                   'military', 'healthcare', 'shop']
    target_cols = [col for col in target_cols if col in gdf.columns]

    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.dropna(subset=target_cols, how='all')
    gdf['feature'] = gdf[target_cols].bfill(axis=1).iloc[:, 0]

    drop_cols = ['fixme', 'note', 'comment', 'description', 'source']
    gdf = gdf.drop(columns=[c for c in drop_cols if c in gdf.columns], errors='ignore')

    return gdf
