# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:42:47 2024

@author: phil_
"""
#%% Step 1
#This code plots the Station fire point with/without landslides and includes the city of LCF for ref. 
#it also gives you the adjusted X and Y Min/Max so you have the LCF study area

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Path to the Mud Creek shapefile directory
mud_creek_shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\MudCreek_Kristin\\MudCreek"

# Read the Mud Creek shapefile
gdf_mud_creek = gpd.read_file(mud_creek_shapefile_path)

# Path to the filtered shapefile directory
filtered_shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\USGS natural hazards data\\US_Landslide_2\\US_Landslide_2\\US_Landslide_2shp\\us_ls_2_poly_filtered.shp"

# Read the filtered shapefile
gdf_filtered = gpd.read_file(filtered_shapefile_path)

# Set the initial extent of the study area in decimal degrees (as you provided)
x_min = -119.7344  # 119° 44' 04" W
x_max = -118.8050  # 118° 48' 18" W
y_min = 34.2300    # 34° 13' 48" N
y_max = 34.6464    # 34° 38' 47" N

# Convert distance from kilometers to degrees
deg_lat_per_km = 1 / 111  # 1 km in degrees latitude ≈ 0.009 degrees
deg_lon_per_km = 1 / (111 * abs(y_min + y_max) / 2 / 111)  # Adjusted for average latitude ≈ 0.009 degrees

# Adjust the bounding box
x_min_adjusted = x_min + 1.5 * deg_lon_per_km  # Move 1 km to the east
x_max_adjusted = x_max - 22.5 * deg_lon_per_km  # Move 5 km to the west
y_min_adjusted = y_min + 13.5 * deg_lat_per_km  # Move 1 km to the north
y_max_adjusted = y_max - 17 * deg_lat_per_km  # Move 1 km to the south

# Print the adjusted Min and Max coordinates for verification
print(f"Adjusted X Min: {x_min_adjusted}, X Max: {x_max_adjusted}")
print(f"Adjusted Y Min: {y_min_adjusted}, Y Max: {y_max_adjusted}")

# Plot the data for Montecito Landslides 2018
fig_montecito = plt.figure(figsize=(9, 9))
ax_montecito = fig_montecito.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_montecito.set_extent([x_min_adjusted, x_max_adjusted, y_min_adjusted, y_max_adjusted])
ax_montecito.add_feature(cfeature.COASTLINE, edgecolor='black')
gdf_filtered.plot(ax=ax_montecito, color='blue', edgecolor='black')
ax_montecito.set_title('Montecito Landslides 2018')

plt.show()
    
#%% Step 2
#Using the adjusted X and Y min/max from the code above, let's create a 30mx30m grid of this area in UTM coordinates

import pandas as pd
import numpy as np
from pyproj import Proj, transform

# Final adjusted Min/Max X and Y coordinates in Lat/Lon
lon_min = -119.69084371657054
lon_max = -119.45834425144172
lat_min = 34.35162162162162
lat_max = 34.493246846846844

# Define UTM Zone 11N (WGS 84 / UTM zone 11N)
proj_latlon = Proj(init='epsg:4326')  # WGS84 Latitude/Longitude
proj_utm = Proj(init='epsg:32611')    # UTM Zone 11N

# Convert bounding box to UTM coordinates
x_min, y_min = transform(proj_latlon, proj_utm, lon_min, lat_min)
x_max, y_max = transform(proj_latlon, proj_utm, lon_max, lat_max)

# Define the grid size in meters
grid_size = 30  # 30 meters x 30 meters

# Calculate the number of rows and columns in the grid
num_cols = int(np.ceil((x_max - x_min) / grid_size))
num_rows = int(np.ceil((y_max - y_min) / grid_size))

# Create lists to hold the grid data
grid_data = []

# Generate the grid data
grid_id = 1
for i in range(num_rows):
    for j in range(num_cols):
        sw_corner_x = x_min + j * grid_size
        sw_corner_y = y_min + i * grid_size
        grid_data.append([grid_id, sw_corner_x, sw_corner_y])
        grid_id += 1

# Convert the grid data into a DataFrame
grid_df = pd.DataFrame(grid_data, columns=['Grid_ID', 'UTM_X', 'UTM_Y'])

# Save the grid data to an Excel file
output_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\1_Mntc_grid system.xlsx"
grid_df.to_excel(output_file_path, index=False)

print(f"Grid system saved to {output_file_path}")


#%% Step 3 
#Using the grid system excel and the USGS KF-Factor data, add an additional column for the KF factor of each cell

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

# Read the "1_Mntc_grid system" Excel file
grid_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\1_Mntc_grid system.xlsx"
grid_df = pd.read_excel(grid_file_path)

# Convert the grid DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(grid_df['UTM_X'], grid_df['UTM_Y'])]
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=geometry, crs="EPSG:32611")

# Load the USGS Soil KF Factor shapefile
shapefile_path = "D:\\Masters\\Thesis Data\\ussoils_18shp\\ussoils_18.shp"
soil_gdf = gpd.read_file(shapefile_path)

# Reproject the soil shapefile to match the CRS of the grid system
soil_gdf = soil_gdf.to_crs("EPSG:32611")

# Function to process and match KF factors
def process_grid(grid_gdf, soil_gdf, grid_size=30):
    # Create polygons for each grid cell
    grid_gdf['geometry'] = grid_gdf.apply(lambda row: box(row['UTM_X'], row['UTM_Y'], row['UTM_X'] + grid_size, row['UTM_Y'] + grid_size), axis=1)

    # Spatial join to intersect grid with KF factors
    grid_intersections = gpd.sjoin(grid_gdf, soil_gdf[['geometry', 'KFFACT']], how='left', predicate='intersects')

    # Use SW coordinate as the cell's location
    grid_intersections['SW_coordinate'] = grid_intersections['geometry'].apply(lambda poly: Point(poly.bounds[0], poly.bounds[1]))

    # Aggregate KF factors for each cell
    grid_kf = grid_intersections.groupby('SW_coordinate')['KFFACT'].mean().reset_index()
    grid_kf.columns = ['SW_coordinate', 'mean_Kf_factor']

    # Convert SW_coordinate back to separate UTM_X and UTM_Y columns
    grid_kf['UTM_X'] = grid_kf['SW_coordinate'].apply(lambda p: p.x)
    grid_kf['UTM_Y'] = grid_kf['SW_coordinate'].apply(lambda p: p.y)
    
    # Merge the KF factor back with the original grid dataframe
    final_grid_df = grid_gdf.merge(grid_kf[['UTM_X', 'UTM_Y', 'mean_Kf_factor']], on=['UTM_X', 'UTM_Y'], how='left')

    return final_grid_df

# Process the grid and match KF factors
result_gdf = process_grid(grid_gdf, soil_gdf)

# Convert the result GeoDataFrame to a regular DataFrame for saving to Excel
result_df = pd.DataFrame(result_gdf.drop(columns='geometry'))

# Save the resulting data to a new Excel file
output_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\2_Mntc_KFfactor.xlsx"
result_df.to_excel(output_file_path, index=False)

print(f"KF factor grid system saved to {output_file_path}")


#%% Step 4
#Visualize the KF Factors by plotting the grid data with KF factors:

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Load the "2_Mntc_KFfactor" Excel file
kf_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\2_Mntc_KFfactor.xlsx"
kf_df = pd.read_excel(kf_file_path)

# Ensure KF factor values are interpreted as numeric
kf_df['mean_Kf_factor'] = pd.to_numeric(kf_df['mean_Kf_factor'])

# Print descriptive statistics for the KF factor values
print(kf_df['mean_Kf_factor'].describe())

# Function to create a polygon for each grid cell based on the SW corner
def create_polygon(utm_x, utm_y, grid_size=30):
    return Polygon([
        (utm_x, utm_y),
        (utm_x + grid_size, utm_y),
        (utm_x + grid_size, utm_y + grid_size),
        (utm_x, utm_y + grid_size),
        (utm_x, utm_y)
    ])

# Convert the DataFrame to a GeoDataFrame
kf_df['geometry'] = kf_df.apply(lambda row: create_polygon(row['UTM_X'], row['UTM_Y']), axis=1)
kf_gdf = gpd.GeoDataFrame(kf_df, geometry='geometry', crs="EPSG:32611")

# Plot the entire dataset
fig, ax = plt.subplots(1, 1, figsize=(20, 20))  # Increased figure size for better visualization

# Remove the boundary plot, just plot the polygons with their colors
kf_gdf.plot(column='mean_Kf_factor', ax=ax, legend=True, cmap='plasma', edgecolor='none')

# Set title and labels
ax.set_title('KF Factors for Each Grid Cell (Full Dataset)', fontsize=16)
ax.set_xlabel('UTM X')
ax.set_ylabel('UTM Y')

# Ensure the plot covers the entire grid area
ax.set_xlim(kf_gdf.total_bounds[0], kf_gdf.total_bounds[2])
ax.set_ylim(kf_gdf.total_bounds[1], kf_gdf.total_bounds[3])

# Show the plot
plt.show()

#%% Step 5
#Calculate and Add dNBR to the Grid System Excel

import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from shapely.geometry import box, Point
import geopandas as gpd

# Paths to the Landsat 8 bands (pre-fire and post-fire)
pre_fire_b5 = "D:\\Masters\\Thesis Data\\USGS EE Burn Ratio\\Thomas Fire\\Before\\LC08_L2SP_042036_20171123_20200902_02_T1_SR_B5.tif"  # NIR
pre_fire_b7 = "D:\\Masters\\Thesis Data\\USGS EE Burn Ratio\\Thomas Fire\\Before\\LC08_L2SP_042036_20171123_20200902_02_T1_SR_B7.tif"  # SWIR
post_fire_b5 = "D:\\Masters\\Thesis Data\\USGS EE Burn Ratio\\Thomas Fire\\After\\LC08_L2SP_042036_20180126_20200902_02_T1_SR_B5.tif"  # NIR
post_fire_b7 = "D:\\Masters\\Thesis Data\\USGS EE Burn Ratio\\Thomas Fire\\After\\LC08_L2SP_042036_20180126_20200902_02_T1_SR_B7.tif"  # SWIR

# Path to the existing Excel file with grid data
grid_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\1_Mntc_grid system.xlsx"
output_file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\3_Mntc_dNBR.xlsx"

# Load the grid system Excel file
grid_df = pd.read_excel(grid_file_path)

# Convert the grid DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(grid_df['UTM_X'], grid_df['UTM_Y'])]
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=geometry, crs="EPSG:32611")

# Function to clip raster data using grid bounds
def clip_raster_to_grid(raster_path, grid_bounds):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [grid_bounds], crop=True)
        return out_image, out_transform

# Get the bounding box of the grid system
grid_bounds = box(grid_gdf.total_bounds[0], grid_gdf.total_bounds[1], grid_gdf.total_bounds[2], grid_gdf.total_bounds[3])

# Clip the rasters
pre_fire_nir_clipped, pre_fire_transform = clip_raster_to_grid(pre_fire_b5, grid_bounds)
pre_fire_swir_clipped, pre_fire_swir_transform = clip_raster_to_grid(pre_fire_b7, grid_bounds)
post_fire_nir_clipped, post_fire_transform = clip_raster_to_grid(post_fire_b5, grid_bounds)
post_fire_swir_clipped, post_fire_swir_transform = clip_raster_to_grid(post_fire_b7, grid_bounds)

# Define a function to sample raster values at grid cell locations
def sample_raster_at_grid(raster_data, transform, grid_gdf):
    sampled_values = []
    for geom in grid_gdf.geometry:
        row, col = ~transform * (geom.x, geom.y)
        row, col = int(row), int(col)
        if (0 <= row < raster_data.shape[1]) and (0 <= col < raster_data.shape[2]):
            sampled_values.append(raster_data[0, row, col])
        else:
            sampled_values.append(np.nan)
    return sampled_values

# Sample raster values at grid locations
grid_gdf['Band5_Before'] = sample_raster_at_grid(pre_fire_nir_clipped, pre_fire_transform, grid_gdf)
grid_gdf['Band7_Before'] = sample_raster_at_grid(pre_fire_swir_clipped, pre_fire_swir_transform, grid_gdf)
grid_gdf['Band5_After'] = sample_raster_at_grid(post_fire_nir_clipped, post_fire_transform, grid_gdf)
grid_gdf['Band7_After'] = sample_raster_at_grid(post_fire_swir_clipped, post_fire_swir_transform, grid_gdf)

# Save the updated grid with band values to a new Excel file
grid_gdf.drop(columns='geometry').to_excel(output_file_path, index=False)

print(f"Band data added and saved to {output_file_path}")


#%% Visualize dNBR data from "4_Mntc_KF&dNBR" excel.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import rasterio
from rasterio.mask import mask
from pyproj import Proj, transform
import numpy as np

# Load the Excel file with dNBR data
file_path = "D:\\Masters\\Thesis Data\\Montecito Data\\4_Mntc_KF&dNBR.xlsx"
df = pd.read_excel(file_path)

# Print the min and max UTM X and Y coordinates (you already did this in Step 1)
print("UTM_X range:", df['UTM_X'].min(), "-", df['UTM_X'].max())
print("UTM_Y range:", df['UTM_Y'].min(), "-", df['UTM_Y'].max())

# Ensure that the DataFrame has the correct CRS (EPSG:32611 for UTM Zone 11N)
crs = "EPSG:32611"

# Function to create a polygon for each grid cell based on the SW corner
def create_polygon(utm_x, utm_y, grid_size=30):
    return Polygon([
        (utm_x, utm_y),
        (utm_x, utm_y + grid_size),
        (utm_x + grid_size, utm_y + grid_size),
        (utm_x + grid_size, utm_y),
        (utm_x, utm_y)
    ])

# Convert the DataFrame to a GeoDataFrame with polygon geometries
df['geometry'] = df.apply(lambda row: create_polygon(row['UTM_X'], row['UTM_Y']), axis=1)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

# Print a few sample geometries to verify correctness
print(gdf.geometry.head())

# Now let's plot the data using GeoPandas without Cartopy
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the geometry and dNBR values
gdf.plot(column='dNBR', cmap='RdYlBu', ax=ax, edgecolor='none')

# Set the title for the plot
ax.set_title('dNBR Values without Cartopy')

# Ensure equal aspect ratio
ax.set_aspect('equal')

# Show the plot
plt.show()
