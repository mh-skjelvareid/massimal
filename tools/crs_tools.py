import geopandas as gpd
import pandas as pd
import shapely


def convert_gcp_to_new_crs(gcp_csv_path_orig,gcp_csv_path_new,delimiter=';',crs_old='EPSG:4326',convert_to_utm=True,crs_new=None):
    """ Read CSV file with ground control points and convert to new CRS 
    
    # Input arguments:
    gcp_csv_path_orig:   Path to text file with GCP points
    gcp_csv_path_new:    Path to file where converted points are saved

    # Keyword arguments:
    delimiter:      Delimiter in CSV file
    crs_old:        EPSG code for original CRS. Default: 'EPSG:4326'
    convert_to_utm: True if points should be converted to UTM
    crs_new:        EPSG code for CRS to convert to, if not UTM. Format: 'EPSG:<code>'

    # Notes
    - Assumes that X and Y coordinates have headers 'X' and 'Y'
    """
    # Validate input
    if not(convert_to_utm) and crs_new is None:
        raise(ValueError('Must specify new CRS if convert_to_utm is False'))
    
    # Create GeoDataFrame from CSV
    data = pd.read_csv(gcp_csv_path_orig,delimiter=delimiter)
    geometry = [shapely.Point(x,y) for (x,y) in zip(data.X,data.Y) ]
    gdf = gpd.GeoDataFrame(data.drop(['X','Y'],axis=1),geometry=geometry,crs=crs_old)

    # Convert to new CSR
    if convert_to_utm: 
        gdf_new = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_new = gdf.to_crs(crs_new)

    # Save to file
    gdf_new.to_csv(gcp_csv_path_new)