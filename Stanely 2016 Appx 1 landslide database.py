# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:04:40 2024

@author: phil_
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from pyproj import Proj, transform

# Read the Excel file
file_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\References\\ofr20161106_appx-1.xlsx"
sheet_name = 'Appendix1_ModelData'  # Update with the actual sheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Filter the data for California (UTM Zone 11)
california_data = df[df['UTM_Zone'] == 11]

# Extract coordinates and landslide data
utm_x = california_data['UTM_X']
utm_y = california_data['UTM_Y']
landslide = california_data['Response']

# Define the bounds in latitude and longitude
lower_lon = -121.922858
lower_lat = 33.596803
upper_lon = -117.839106
upper_lat = 36.314017

# Convert latitude and longitude bounds to UTM
in_proj = Proj(init='epsg:4326', preserve_units=False)  # WGS84 coordinate system for latitude and longitude
out_proj = Proj(init='epsg:32611', preserve_units=False)  # UTM Zone 11 coordinate system
lower_x, lower_y = transform(in_proj, out_proj, lower_lon, lower_lat)
upper_x, upper_y = transform(in_proj, out_proj, upper_lon, upper_lat)

# Create a scatter plot if there are data points within the specified bounds
if not (utm_x.min() > upper_x or utm_x.max() < lower_x or utm_y.min() > upper_y or utm_y.max() < lower_y):
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.UTM(zone=11))
    ax.coastlines(resolution='10m')

    # Plot the wildfire data
    scatter = ax.scatter(utm_x, utm_y, c=landslide, cmap='RdYlGn', transform=ccrs.UTM(zone=11))

    # Add a colorbar to show landslide occurrence
    cb = plt.colorbar(scatter, label='Landslide (0 = No, 1 = Yes)')
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['No', 'Yes'])

    # Set x and y axis limits
    ax.set_xlim([lower_x, upper_x])
    ax.set_ylim([lower_y, upper_y])

    plt.title('California Wildfire Data')
    plt.xlabel('UTM X Coordinate')
    plt.ylabel('UTM Y Coordinate')

    plt.savefig('plot.png')
    plt.show()
else:
    print("No data points within the specified bounds.")