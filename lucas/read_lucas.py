import pandas as pd
import geopandas as gpd


def read_lucas(
    shapefile_path: str = "LUCAS_2018_Copernicus_polygons.shp",
    csv_path: str = "LUCAS_2018_Copernicus_attributes.csv",
) -> gpd.GeoDataFrame:
    lucas_gdf = gpd.read_file(shapefile_path)
    attributes_df = pd.read_csv(csv_path)
    lucas_gdf["POINT_ID"] = lucas_gdf["POINT_ID"].astype(int)
    attributes_df["POINT_ID"] = attributes_df["POINT_ID"].astype(int)
    return lucas_gdf.merge(attributes_df, on="POINT_ID")


def get_lucas_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = gdf.total_bounds
    return (bounds[1], bounds[0], bounds[3], bounds[2])  # (miny, minx, maxy, maxx)
