# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:52:19 2024

@author: phil_
"""

import geopandas as gpd
import pandas as pd

# Path to the original shapefile directory
original_shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\USGS natural hazards data\\US_Landslide_2\\US_Landslide_2\\US_Landslide_2shp\\us_ls_2_poly.shp"

# Read the original shapefile
gdf_original = gpd.read_file(original_shapefile_path)

# Convert the Date column to datetime
gdf_original['Date'] = pd.to_datetime(gdf_original['Date'], errors='coerce')

# Filter the data by region (Southern California) and date range (after 2009)
lower_x = -121.922858
lower_y = 33.596803
upper_x = -117.839106
upper_y = 36.314017

gdf_filtered = gdf_original.cx[lower_x:upper_x, lower_y:upper_y]
gdf_filtered = gdf_filtered[gdf_filtered['Date'] >= pd.to_datetime('2009-01-01', errors='coerce')]

# Convert the Date column back to string format
gdf_filtered['Date'] = gdf_filtered['Date'].dt.strftime('%Y-%m-%d')

# Path to save the new shapefile
filtered_shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\USGS natural hazards data\\US_Landslide_2\\US_Landslide_2\\US_Landslide_2shp\\us_ls_2_poly_filtered.shp"

# Save the filtered data to a new shapefile
gdf_filtered.to_file(filtered_shapefile_path)

print("Filtered shapefile saved successfully!")

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.ops import unary_union
from shapely.geometry import Point

# Path to the filtered shapefile directory
filtered_shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\USGS natural hazards data\\US_Landslide_2\\US_Landslide_2\\US_Landslide_2shp\\us_ls_2_poly_filtered.shp"

# Read the filtered shapefile
gdf_filtered = gpd.read_file(filtered_shapefile_path)

# Define x and y edges
lower_x = -121.922858
lower_y = 33.596803
upper_x = -117.839106
upper_y = 36.314017

# Plot the filtered data within the specified bounds
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot coastline
ax.add_feature(cfeature.COASTLINE, edgecolor='black')

# Plot filtered landslide data
gdf_filtered.plot(ax=ax, color='blue', edgecolor='black')

# Buffer distance to merge nearby points (1km)
buffer_distance = 1000  # in meters

# Merge nearby points based on buffer distance
gdf_filtered_merged = gdf_filtered.copy()
gdf_filtered_merged['geometry'] = gdf_filtered_merged.buffer(buffer_distance)

# Define function to calculate new positions for callouts
def calculate_callout_position(geom, offset=0.01):
    centroid = geom.centroid
    return Point(centroid.x + offset, centroid.y + offset)

# Add callouts with dates and arrows
for idx, row in gdf_filtered.iterrows():
    geom = gdf_filtered_merged.loc[idx, 'geometry']
    date = row['Date']
    
    # Calculate new position for callout
    callout_pos = calculate_callout_position(geom)
    
    # Plot arrow from original position to callout position
    ax.annotate('', xy=(callout_pos.x, callout_pos.y), xytext=(geom.centroid.x, geom.centroid.y),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    # Plot text callout with date
    ax.text(callout_pos.x, callout_pos.y, str(date), fontsize=8, color='black', ha='center', va='center')

# Set plot title and labels
plt.title('Landslide Polygons in Southern California after 2009 with Coastline and Date Callouts')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Set axis limits to focus on the specified area
ax.set_xlim([lower_x, upper_x])
ax.set_ylim([lower_y, upper_y])

# Show the plot
plt.show()

#%%
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

# Define coordinates for the Montecito landslides
montecito_coords = (-119.6457 + 0.015, 34.4367)  # Shifted by 2km to the east

# Plot the data for Montecito Landslides 2018
fig_montecito = plt.figure(figsize=(9, 9))
ax_montecito = fig_montecito.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_montecito.set_extent([montecito_coords[0] - 0.035, montecito_coords[0] + 0.045, 
                         montecito_coords[1] - 0.035, montecito_coords[1] + 0.035])
ax_montecito.add_feature(cfeature.COASTLINE, edgecolor='black')
gdf_filtered.plot(ax=ax_montecito, color='blue', edgecolor='black')
ax_montecito.set_title('Montecito Landslides 2018')

plt.show()

# # Define coordinates for the Mud Creek Landslide
# mud_creek_coords = (-121.430991, 35.865225)

# # Plot the data for Mud Creek Landslide
# fig_mud_creek = plt.figure(figsize=(6, 6))
# ax_mud_creek = fig_mud_creek.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# ax_mud_creek.set_extent([mud_creek_coords[0] - 0.015, mud_creek_coords[0] + 0.015, 
#                          mud_creek_coords[1] - 0.015, mud_creek_coords[1] + 0.015])
# ax_mud_creek.add_feature(cfeature.COASTLINE, edgecolor='black')
# gdf_mud_creek.plot(ax=ax_mud_creek, color='green', edgecolor='black')
# ax_mud_creek.set_title('Mud Creek Landslide')

#%%

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

# Calculate the bounding box of the filtered shapefile data
bounds = gdf_filtered.total_bounds  # returns [xmin, ymin, xmax, ymax]
x_min, y_min, x_max, y_max = bounds

# Print the original Min and Max coordinates
print(f"Original X Min: {x_min}, X Max: {x_max}")
print(f"Original Y Min: {y_min}, Y Max: {y_max}")

# Convert buffer from meters to degrees
buffer_deg_lat = 0.5 / 111  # Approximately 0.5 km in degrees latitude
buffer_deg_lon = 0.5 / (111 * abs(y_min + y_max) / 2 / 111)  # Adjusted by average latitude

# Apply the buffer to the X and Y coordinates
x_min_adjusted = x_min - buffer_deg_lon
x_max_adjusted = x_max + buffer_deg_lon
y_min_adjusted = y_min - buffer_deg_lat
y_max_adjusted = y_max + buffer_deg_lat

# Print the adjusted Min and Max coordinates
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
