# This script reads all csv files in the directories "vector" and "search" in the directory "raw_data", calculates the
# transformed x and y coordinates with the origin at the release site, and saves the data to a csv file called "all_data.csv".

import os
import pandas as pd
import numpy as np
import json

# Define input and output directories:
input_directory = "raw_data/"
output_directory = "tables/"
output_filename = "all_data.csv"

# Walk through the input directory and get all subdirectories and files:
release_sites = [x[0] for x in os.walk(input_directory)]
release_sites = [x for x in release_sites if x.endswith("vector") or x.endswith("search")]
print(release_sites)

# Create the output directory if it does not exist:
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the json file with the coordinates of the release sites and the feeder:
with open("landmark_coordinates.json") as file:
    params = json.load(file)
for key, value in params.items():
    globals()[key] = value

# Read each csv file in each directory using pandas and append it horizontally to a new dataframe. In addition,
# add a column with the release site name ("R1", "R2", etc.) and a column with either "vector" or "search"
# and the name of the original file before the second "-":

df = pd.DataFrame()
for release_site in release_sites:
    # Get the release site name from the path:
    release_site_name = release_site.split("/")[-2]
    # Get the type of data (vector or search) from the path:
    flight_type = release_site.split("/")[-1]
    # Add all files in the directory to a list:
    files = os.listdir(release_site)

    # Only keep files that end with ".csv", print discarded files:
    files = [x for x in files if x.endswith(".csv")]
    # Print files that do not end with ".csv" and ask the user if we should continue. In red:
    for file in os.listdir(release_site):
        print(file)
        if not file.endswith(".csv"):
            print("\033[91m" + "Non CSV: " + file + "\033[0m")
            quit("figure this out")

    # Read each file in the list using pandas and append it vertically to a new dataframe:
    df_temp = pd.DataFrame()
    for file in files:
        # Read the file using pandas:
        df_temp2 = pd.read_csv(release_site + "/" + file, header=None)
        # Give names to the original columns:
        df_temp2.columns = ["x_orig", "y_orig", "t"]

        # Correct the xy coordinate for R1 and R2 by subtracting the release coordinates x0 and y0
        # and setting the current release site coordinates as the origin (0, 0):
        # Define x0 and y0 based on the release site name:
        if release_site_name == "R1":
            x0 = R1_release_x
            y0 = R1_release_y
        elif release_site_name == "R2":
            x0 = R2_release_x
            y0 = R2_release_y
        else:
            x0 = 0
            y0 = 0

       # replace nan with -1
        df_temp2 = df_temp2.fillna(-1)

        # Add a column with the transformed x coordinate:
        df_temp2["x"] = df_temp2["x_orig"] - x0
        # Add a column with the transformed y coordinate:
        df_temp2["y"] = df_temp2["y_orig"] - y0

        # Add a column called "dt", calculate the difference between consecutive t values:
        df_temp2["dt"] = np.append([np.nan], np.diff(df_temp2["t"]))

        # Calculate distances
        df_temp2["distance_from_previous_position"] = np.append([np.nan], np.sqrt(np.diff(df_temp2["x"])**2 + np.diff(df_temp2["y"])**2))   
        df_temp2["distance_from_release_site"] = np.sqrt(df_temp2["x"]**2 + df_temp2["y"]**2)

        # Calculate speeds from one position to the next (prone to errors):
        df_temp2["dv"] = df_temp2["distance_from_previous_position"] / df_temp2["dt"]

        # Disable for now:
        if False:
            df_temp2["distance_to_feeder"] = np.sqrt((df_temp2["x_orig"] - feeder_x)**2 + (df_temp2["y_orig"] - feeder_y)**2)
            df_temp2["distance_to_virtual_feeder"] = np.sqrt((df_temp2["x"] - feeder_x)**2 + (df_temp2["y"] - feeder_y)**2)

        # Calculate bearings:
        df_temp2["bearing_from_previous_position"] = np.degrees(np.arctan2(df_temp2["x"].diff(), df_temp2["y"].diff()))
        df_temp2["bearing_from_release_site"] = np.degrees(np.arctan2(df_temp2["x"], df_temp2["y"]))

        # Add 360 degrees to negative angles so that the bearing is between 0 and 360 degrees:
        df_temp2.loc[df_temp2["bearing_from_previous_position"] < 0, "bearing_from_previous_position"] += 360
        df_temp2.loc[df_temp2["bearing_from_release_site"] < 0, "bearing_from_release_site"] += 360

        # Reset the index:
        df_temp2 = df_temp2.reset_index(drop=True)

        # Add a column with the bee name:
        bee = file.split("-")[0] + "-" + file.split("-")[1]
        df_temp2.insert(0, "Bee", bee)
   
        # Concatinate the dataframe to df_temp vertically:
        df_temp = pd.concat([df_temp, df_temp2], axis=0)
        print("Done processing", file)

    # Add a column with the release site name:
    df_temp["release_site"] = release_site_name
    # Add a column with the type of data:
    df_temp["flight_type"] = flight_type

    # Concatinate the dataframe to df vertically:
    df = pd.concat([df, df_temp], axis=0)

# Save the dataframe to a csv file:
df = df.reset_index(drop=True)
df.to_csv(output_directory + output_filename, index=False)
print("\nDone processing all data!")
