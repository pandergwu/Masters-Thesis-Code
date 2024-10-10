# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:42:47 2024

@author: phil_
"""
#%% This code plots the Station fire point with/without landslides and includes the city of LCF for ref. 
#it also gives you the adjusted X and Y Min/Max so you have the LCF study area
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import numpy as np

# Function to convert latitude and longitude to UTM coordinates
def latlon_to_utm(lat, lon, utm_zone):
    transformer = Transformer.from_crs("epsg:4326", f"epsg:326{utm_zone:02d}")
    utm_x, utm_y = transformer.transform(lat, lon)
    return utm_x, utm_y, utm_zone

# File path and sheet name
file_path = "C:\\Users\\phil_\\OneDrive\\Desktop\\Master's Degree\\Thesis Data\\References\\ofr20161106_appx-1.xlsx"
sheet_name = "Appendix1_ModelData"

# Read Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Check for NaN values in the DataFrame
nan_values = df.isnull().sum().sum()

# Check for Inf values only in numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
inf_values = np.isinf(numeric_df).sum().sum()

# Total number of NaN or Inf values
nan_inf_values = nan_values + inf_values

print("\nNumber of NaN values in DataFrame:", nan_values)
print("Number of Inf values in DataFrame:", inf_values)
print("Total number of NaN or Inf values in DataFrame:", nan_inf_values)

# Filter data for points related to the Station fire
# Remove rows where 'Fire Name' column has NaN values
df = df.dropna(subset=['Fire Name'])
station_fire_df = df[df['Fire Name'].str.contains("Station", na=False)]

# Extract relevant columns
station_fire_df = station_fire_df[['UTM_Zone', 'UTM_X', 'UTM_Y', 'Response']]

# Define colors based on Response column
colors = {0: 'black', 1: 'red'}

# Create GeoDataFrame from DataFrame
geometry = [Point(xy) for xy in zip(station_fire_df['UTM_X'], station_fire_df['UTM_Y'])]
gdf = gpd.GeoDataFrame(station_fire_df, geometry=geometry, crs="EPSG:26911")

# Ensure there are no invalid UTM coordinates
gdf = gdf.replace([np.inf, -np.inf], np.nan)
gdf = gdf.dropna(subset=['UTM_X', 'UTM_Y'])

# Print out a sample of the cleaned data for debugging
print(gdf.head())

# Convert city coordinates from latitude and longitude to UTM
cities = {
    'La Canada-Flintridge': (-118.2009, 34.2069),
}
# Use UTM zone 11 explicitly for the city conversion
city_coordinates_utm = {city: latlon_to_utm(lat, lon, 11) for city, (lon, lat) in cities.items()}

# Print out the UTM coordinates of the city for debugging
print("La Canada-Flintridge UTM coordinates:", city_coordinates_utm['La Canada-Flintridge'])

# Adjust southern padding boundary
la_canada_flintridge_utm = latlon_to_utm(cities['La Canada-Flintridge'][1], cities['La Canada-Flintridge'][0], 11)

# Verify the calculated UTM coordinates
print("La Canada-Flintridge UTM (separate):", la_canada_flintridge_utm)

padding_south = 50  # 50 meters south of La Canada Flintridge
x_min, x_max = gdf['UTM_X'].min(), gdf['UTM_X'].max()
y_min, y_max = gdf['UTM_Y'].min(), gdf['UTM_Y'].max()

# Verify the initial axis limits
print("Initial x_min:", x_min)
print("Initial x_max:", x_max)
print("Initial y_min:", y_min)
print("Initial y_max:", y_max)

# Adjust y_max based on southern padding
y_max = max(y_max, la_canada_flintridge_utm[1] + padding_south)

# Add extra padding to the east, north, and west
padding_factor = 0.1  # 10% padding
x_range = x_max - x_min
y_range = y_max - y_min
x_min -= padding_factor * x_range
x_max += padding_factor * x_range
y_min -= padding_factor * y_range
y_max += padding_factor * y_range

# Print out axis limits for debugging
print("Adjusted x_min:", x_min)
print("Adjusted x_max:", x_max)
print("Adjusted y_min:", y_min)
print("Adjusted y_max:", y_max)

# Check for NaN or Inf in axis limits
invalid_values = [val for val in [x_min, x_max, y_min, y_max] if np.isnan(val) or np.isinf(val)]
if invalid_values:
    print("Invalid axis limits. Check data for NaN or Inf values.")
    print("Invalid values:", invalid_values)
else:
    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(11))

    # Plot coastline
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    # Plot cities using UTM coordinates only
    for city, (utm_x, utm_y, utm_zone) in city_coordinates_utm.items():
        ax.plot(utm_x, utm_y, '*', color='black', markersize=10, zorder=1)
        ax.text(utm_x, utm_y, city, fontsize=8, color='black', ha='left', va='bottom')

    # Plot points
    for idx, row in gdf.iterrows():
        color = colors[row['Response']]
        ax.scatter(row['UTM_X'], row['UTM_Y'], color=color)

    # Set axis limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    # Set plot title and labels
    ax.set_title('Geometry of Points Related to Station Fire')
    ax.set_xlabel('UTM X')
    ax.set_ylabel('UTM Y')

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='No Landslide', markerfacecolor='black', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='Landslide', markerfacecolor='red', markersize=10)]
    ax.legend(handles=legend_elements)

    # Show the plot
    plt.show()
    
#%% Using the adjusted X and Y min/max from the code above let's try to create a 30mx30m grid of this area

import pandas as pd
import numpy as np

# Define the grid size
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
output_file_path = "D:\\Masters\\Thesis Data\\LCF Data\\1_LCF_grid system.xlsx"
grid_df.to_excel(output_file_path, index=False)

print(f"Grid system saved to {output_file_path}")

#%% Using this grid system excel and the USGS KF-Factor data spreadsheet, add an additional column for the 
#KF factor of each cell

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np

# Read the "LCF_grid system" Excel file
grid_file_path = "D:\\Masters\\Thesis Data\\LCF Data\\1_LCF_grid system.xlsx"
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

    # Spatial join to intersect grid with Kf factors
    grid_intersections = gpd.sjoin(grid_gdf, soil_gdf[['geometry', 'KFFACT']], how='left', predicate='intersects')

    # Use SW coordinate as the cell's location
    grid_intersections['SW_coordinate'] = grid_intersections['geometry'].apply(lambda poly: Point(poly.bounds[0], poly.bounds[1]))

    # Aggregate Kf factors for each cell
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
output_file_path = "D:\\Masters\\Thesis Data\\LCF Data\\2_LCF_KFfactor.xlsx"
result_df.to_excel(output_file_path, index=False)

print(f"KF factor grid system saved to {output_file_path}")

#%% Visualize "a meaningful section" of the KF Factors by plotting the grid data w/ KF factors: 

# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon

# # Load the "2_LCF_KFfactor" Excel file
# kf_file_path = "D:\\Masters\\Thesis Data\\LCF Data\\2_LCF_KFfactor.xlsx"
# kf_df = pd.read_excel(kf_file_path)

# # Ensure KF factor values are interpreted as numeric
# kf_df['mean_Kf_factor'] = pd.to_numeric(kf_df['mean_Kf_factor'])

# # Print a sample of the data for debugging
# print(kf_df.head())

# # Check for non-zero KF factor values
# print("Number of non-zero KF factor values:", (kf_df['mean_Kf_factor'] > 0).sum())
# print(kf_df['mean_Kf_factor'].describe())

# # Function to create a polygon for each grid cell based on the SW corner
# def create_polygon(utm_x, utm_y, grid_size=30):
#     return Polygon([
#         (utm_x, utm_y),
#         (utm_x + grid_size, utm_y),
#         (utm_x + grid_size, utm_y + grid_size),
#         (utm_x, utm_y + grid_size),
#         (utm_x, utm_y)
#     ])

# # Convert the DataFrame to a GeoDataFrame
# kf_df['geometry'] = kf_df.apply(lambda row: create_polygon(row['UTM_X'], row['UTM_Y']), axis=1)
# kf_gdf = gpd.GeoDataFrame(kf_df, geometry='geometry', crs="EPSG:32611")

# # Print a sample of the GeoDataFrame for debugging
# print(kf_gdf.head())

# # Select a subset of the data for visualization
# # Let's select a meaningful subset by limiting both UTM_X and UTM_Y
# subset_condition = (kf_df['UTM_X'] < 380000) & (kf_df['UTM_Y'] < 3788000)
# kf_gdf_subset = kf_gdf[subset_condition]

# # Ensure that we have enough data points in the subset
# print(f"Number of cells in the subset: {len(kf_gdf_subset)}")

# # Plot the subset of the KF factors for easier debugging
# fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# kf_gdf_subset.boundary.plot(ax=ax, linewidth=0.5, color='black')

# # Try a different colormap and ensure the plot scales correctly
# kf_gdf_subset.plot(column='mean_Kf_factor', ax=ax, legend=True, cmap='plasma', edgecolor='black')

# # Plot city for reference
# city_utm_coords = {'La Canada-Flintridge': (389366.08254841727, 3785748.8919543694)}
# for city, (utm_x, utm_y) in city_utm_coords.items():
#     ax.plot(utm_x, utm_y, '*', color='red', markersize=15, zorder=5)
#     ax.text(utm_x, utm_y, city, fontsize=12, color='red', ha='left', va='bottom')

# # Set title and labels
# ax.set_title('KF Factors for Each Grid Cell (Subset)', fontsize=16)
# ax.set_xlabel('UTM X')
# ax.set_ylabel('UTM Y')

# # Ensure the plot covers the subset area
# ax.set_xlim(kf_gdf_subset.total_bounds[0], kf_gdf_subset.total_bounds[2])
# ax.set_ylim(kf_gdf_subset.total_bounds[1], kf_gdf_subset.total_bounds[3])

# # Show the plot
# plt.show()

#%%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Load the "2_LCF_KFfactor" Excel file
kf_file_path = "D:\\Masters\\Thesis Data\\LCF Data\\2_LCF_KFfactor.xlsx"
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

# Plot city for reference
city_utm_coords = {'La Canada-Flintridge': (389366.08254841727, 3785748.8919543694)}
print(f"City coordinates: {city_utm_coords}")

for city, (utm_x, utm_y) in city_utm_coords.items():
    if kf_gdf.total_bounds[0] <= utm_x <= kf_gdf.total_bounds[2] and kf_gdf.total_bounds[1] <= utm_y <= kf_gdf.total_bounds[3]:
        ax.plot(utm_x, utm_y, '*', color='red', markersize=15, zorder=5)
        ax.text(utm_x, utm_y, city, fontsize=12, color='red', ha='left', va='bottom')
    else:
        print(f"City {city} is outside the plot bounds.")

# Set title and labels
ax.set_title('KF Factors for Each Grid Cell (Full Dataset)', fontsize=16)
ax.set_xlabel('UTM X')
ax.set_ylabel('UTM Y')

# Ensure the plot covers the entire grid area
ax.set_xlim(kf_gdf.total_bounds[0], kf_gdf.total_bounds[2])
ax.set_ylim(kf_gdf.total_bounds[1], kf_gdf.total_bounds[3])

# Show the plot
plt.show()


