import sys
sys.path.insert(0, '/home/michael/project_khthon')
sys.path.insert(0, '/home/michael/project_khthon/open_street_map')

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.ops import transform as shapely_transform
from rasterio.features import rasterize
from pyproj import Transformer
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer

from get_data_from_eo import get_image
from get_osm_boxes import get_osm_features

TIFF_CACHE = Path('/home/michael/project_khthon/open_street_map/tiff_cache')
TIFF_CACHE.mkdir(parents=True, exist_ok=True)


def label_sentinel2_with_osm(
    img,
    osm_gdf: gpd.GeoDataFrame,
    label_col: str = 'feature',
    no_object_label: str = 'no_object',
) -> np.ndarray:
    img_crs = img.rio.crs
    nrows, ncols = img.shape[1], img.shape[2]
    transform = img.rio.transform()

    valid = osm_gdf[label_col].notna()
    unique_vals = osm_gdf.loc[valid, label_col].unique()
    val_to_int = {v: i + 1 for i, v in enumerate(unique_vals)}

    to_img_crs = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)

    shapes = []
    for _, row in osm_gdf[valid].iterrows():
        geom_img = shapely_transform(
            lambda x, y: to_img_crs.transform(x, y), row.geometry
        )
        shapes.append((geom_img, val_to_int[row[label_col]]))

    int_mask = rasterize(
        shapes,
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    str_mask = np.full((nrows, ncols), no_object_label, dtype=object)
    for val, code in val_to_int.items():
        str_mask[int_mask == code] = val

    return str_mask


def build_tabular_dataset(
    img,
    label_mask: np.ndarray,
    no_object_label: str = 'no_object',
) -> pd.DataFrame:
    bands = img.squeeze().values
    n_bands = bands.shape[0]

    nrows, ncols = bands.shape[1], bands.shape[2]

    pixel_data = bands.reshape(n_bands, -1).T
    labels_flat = label_mask.ravel()

    band_names = [f'B{str(i).zfill(2)}' for i in range(1, n_bands + 1)]
    if hasattr(img, 'band') and hasattr(img.band, 'values'):
        band_names = [str(b) for b in img.band.values]

    df = pd.DataFrame(pixel_data, columns=band_names)
    df['label'] = labels_flat

    return df


def filter_rare_classes(df: pd.DataFrame, label_col: str = 'label', min_samples: int = 10):
    counts = df[label_col].value_counts()
    keep = counts[counts >= min_samples].index
    dropped = len(counts) - len(keep)
    if dropped:
        print(f"  Dropped {dropped} class(es) with <{min_samples} samples: "
              f"{list(counts[counts < min_samples].index)}")
    return df[df[label_col].isin(keep)].copy()


def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])


def _tiff_cache_path(lat: float, lon: float, height: int, width: int,
                     days: int, time: datetime | None) -> Path:
    ts = time.isoformat() if time else 'None'
    name = f'{lat:.4f}_{lon:.4f}_{height}_{width}_{days}_{ts}.tiff'
    return TIFF_CACHE / name


def dataset_for_location(
    lat: float,
    lon: float,
    height: int = 5000,
    width: int = 5000,
    osm_dist: int = 5000,
    days: int = 30,
    time: datetime | None = None,
    samples_per_class: int | None = 5000,
) -> pd.DataFrame:
    cache_path = _tiff_cache_path(lat, lon, height, width, days, time)
    if cache_path.exists():
        print(f"  Loading cached Sentinel-2 image from {cache_path.name}")
        img = rioxarray.open_rasterio(cache_path)
    else:
        print(f"  Fetching Sentinel-2 image for ({lat:.4f}, {lon:.4f})...")
        img = get_image(lat, lon, height, width, days=days, time=time)
        print(f"  Caching to {cache_path.name}")
        img.rio.to_raster(cache_path)

    print(f"  Fetching OSM features...")
    osm_gdf = get_osm_features(lat, lon, osm_dist)

    print(f"  Rasterizing OSM labels...")
    label_mask = label_sentinel2_with_osm(img, osm_gdf)

    print(f"  Building tabular dataset...")
    df = build_tabular_dataset(img, label_mask)

    if samples_per_class is not None:
        print(f"  Downsampling to ≤{samples_per_class} pixels per class...")
        df = (
            df.groupby('label', group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), samples_per_class), random_state=42))
        )

    df['location_lat'] = lat
    df['location_lon'] = lon

    print(f"  -> {len(df)} pixels, {df['label'].nunique()} classes")
    return df


def pipeline_from_locations(
    locations: list[tuple[float, float]],
    test_size: float = 0.2,
    random_state: int = 42,
    height: int = 5000,
    width: int = 5000,
    osm_dist: int = 5000,
    days: int = 30,
    time: datetime | None = None,
    samples_per_class: int | None = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    print("=== PROCESSING LOCATIONS ===")
    dfs = []
    for i, (lat, lon) in enumerate(locations):
        print(f"[{i+1}/{len(locations)}]")
        try:
            df = dataset_for_location(lat, lon, height, width, osm_dist, days, time,
                                      samples_per_class=samples_per_class)
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    df = pd.concat(dfs, ignore_index=True)
    df = filter_rare_classes(df, label_col='label', min_samples=10)
    print(f"\nCombined data: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    pipe = build_pipeline()
    pipe.fit(train_df.drop(columns='label').values, train_df['label'].values)
    known_classes = set(pipe.classes_)

    unseen = ~test_df['label'].isin(known_classes)
    if unseen.any():
        n = unseen.sum()
        print(f"  Mapping {n} test pixels with unseen labels -> 'no_object'")
        test_df = test_df.copy()
        test_df.loc[unseen, 'label'] = 'no_object'

    print(f"\nTrain set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print(f"Test class distribution:\n{test_df['label'].value_counts()}")

    return train_df, test_df, pipe


if __name__ == '__main__':
    out = Path('/home/michael/project_khthon/open_street_map/evaluation_output')
    out.mkdir(parents=True, exist_ok=True)

    import re
    graves_df = pd.read_excel('/home/michael/project_khthon/grave_inference/AI Grave Dataset.xlsx')
    def dms_to_dd(dms):
        m = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NS])\s*(\d+)°(\d+)'([\d.]+)\"([EW])", str(dms))
        if not m:
            return None, None
        lat = float(m[1]) + float(m[2])/60 + float(m[3])/3600
        lon = float(m[5]) + float(m[6])/60 + float(m[7])/3600
        if m[4] == 'S': lat = -lat
        if m[8] == 'W': lon = -lon
        return lat, lon

    locations = []
    for c in graves_df['Coordinates'][:20]:
        lat, lon = dms_to_dd(c)
        if lat is not None:
            locations.append((lat, lon))

    print(f"Loaded {len(locations)} locations from AI Grave Dataset")

    train_df, test_df, pipe = pipeline_from_locations(
        locations, height=5000, width=5000
    )

    model_path = out / 'model.joblib'
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")

    classes = pipe.classes_
    present = test_df['label'].isin(classes)
    if not present.all():
        n_removed = (~present).sum()
        print(f"Removing {n_removed} test rows with labels not in model classes")
        test_df = test_df[present].copy()

    if test_df.empty:
        print("No test samples remaining after filtering. Skipping evaluation.")
    else:
        X_test = test_df.drop(columns='label').values
        y_test = test_df['label'].values
        y_pred = pipe.predict(X_test)

        # --- classification report ---
        report = classification_report(y_test, y_pred, labels=classes, target_names=classes, zero_division=0)
        with open(out / 'classification_report.txt', 'w') as f:
            f.write(report)
        print("Saved classification report.")

        # --- confusion matrix ---
        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5),) * 2)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=classes, ax=ax,
                                                cmap='Blues', xticks_rotation=45)
        fig.tight_layout()
        fig.savefig(out / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
        print("Saved confusion matrix.")

        # --- ROC curves (one-vs-rest) ---
        lb = LabelBinarizer()
        lb.fit(classes)
        y_test_bin = lb.transform(y_test)
        y_prob = pipe.predict_proba(X_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        fig.savefig(out / 'roc_curves.png', dpi=150)
        plt.close(fig)
        print("Saved ROC curves.")

        # --- Average Precision curves ---
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
            ax.plot(recall, precision, label=f'{cls} (AP = {ap:.3f})')

        micro_prec, micro_rec, _ = precision_recall_curve(y_test_bin.ravel(), y_prob.ravel())
        micro_ap = average_precision_score(y_test_bin, y_prob, average='micro')
        ax.plot(micro_rec, micro_prec, 'k--', linewidth=2,
                label=f'micro-average (AP = {micro_ap:.3f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        fig.savefig(out / 'precision_recall_curves.png', dpi=150)
        plt.close(fig)
        print("Saved precision-recall curves.")

        print(f"\nAll outputs saved to {out}/")
