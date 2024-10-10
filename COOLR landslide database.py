# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:03 2024

@author: phil_
"""

import geopandas as gpd
import matplotlib.pyplot as plt

# Path to the shapefile directory
shapefile_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\COOLR data\\shapefiles\\nasa_coolr_reports_poly.shp"

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Get the bounding box of the data
bbox = gdf.total_bounds

# Plot the original data
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='blue', edgecolor='black')

# Set plot title and labels
plt.title('Landslide Polygons in Southern California/Greater Los Angeles Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Set axis limits to cover the bounding box of the data
ax.set_xlim([bbox[0], bbox[2]])
ax.set_ylim([bbox[1], bbox[3]])

# Show the plot
plt.show()

