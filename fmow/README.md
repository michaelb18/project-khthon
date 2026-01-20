# fMoW-Sentinel metadata

The fMoW-Sentinel dataset consists of Sentinel-2 satellite image time series corresponding to locations from the Functional Map of the World (fMoW) dataset.   

## Dataset information
Each image contains 13 bands, corresponding to the 13 bands of Sentinel-2, in order of increasing wavelength (B1, B2, B3, B4, B5, B6, B7, B8A, B8B, B9, B10, B11, B12). B4, B3, and B2 correspond to red, green, and blue, respectively. Some images that failed to download have a single null band; these images are filtered out from the metadata csv files.  

The time series for each location is constructed as follows: First, all fMoW image times before 2015-07-01 (close to the start date of Sentinel-2) are removed. If the location has no fMoW images after this filtering, the location is removed. Then, to increase the number of datapoints per location, for each of the 9 6-month intervals from 2015-07-01 to 2019-12-31, a new timestamp is placed at the midpoint of the interval if there is no fMoW image in that interval, effectively making each time series at least 9 images in length; since some images fail to download, time series may be somewhat shorter after filtering them out. For each of the times within the time series, a Sentinel-2 cloud composite is constructed over the interval from 45 days before to 45 days after the image time.

## Metadata fields
- `category`: fMoW category of the image
- `location_id`: Numerical id for which location the image belongs to (within its category), taken directly from fMoW
- `image_id`: Numerical id for the image within its location. `image_id` >= 100 indicates that the image has no corresponding fMoW image, while any `image_id` < 100 is taken directly from fMoW
- `timestamp`: Center time of the image composite interval, formatted as `YYYY-MM-DDThh:mm:ssZ` (where T and Z are the literal letters)
- `polygon`: Set of coordinates specifying the lat/long of the corners of the image location

## Splits
- `train.csv`: Corresponds to locations in the fMoW train set
- `val.csv`: Corresponds to locations in the fMoW validation set
- `test_gt.csv`: Corresponds to locations in the fMoW test set

## Image path
The path to a given file is `<split>/<category>/<category_image_id>/<category>_<image_id>_<location_id>.tif`. For example, `train/airport/airport_0/airport_0_6.tif`.

## Licensing
The fMoW-Sentinel dataset is derived from two data sources with their own licenses: The Functional Map of the World Challenge Public License (https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE) applies to the locations and categories of the images in the dataset (i.e. the data in the metadata CSV files), while the Sentinel-2 License (https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf) applies to the images themselves.