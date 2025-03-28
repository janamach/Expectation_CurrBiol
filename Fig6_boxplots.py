import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import scipy.stats as stats
import numpy as np

input_file = "tables/all_data.csv"

figure_outname = "figures/Fig6_kruskal.png"
p_value_style = "star"

# Set the following variables to True if you want to generate new csv files:
save_output_to_csv = True
output_csv_name = "tables/Fig6_search_flight_characteristics.csv"
save_variance_table = True
variance_csv_name = "tables/Fig6_search_flight_variance.csv"
stats_csv_name = "tables/Fig6_search_flight_stats.csv"

# Read input file into a dataframe:
df = pd.read_csv(input_file)

# Only take search flight data:
df = df[df["flight_type"] == "search"]

# Create an empty dataframe to store the flight characteristics:
df_output = pd.DataFrame()
df_stats = pd.DataFrame()

# Iterate through bees and calculate the search flight characteristics:
for bee, bdf in df.groupby("Bee"):
    print("Search flight performed", bee)
    # Calculate the total distance of when flight_type is search:
    search_length = bdf[bdf["flight_type"] == "search"]["distance_from_previous_position"].sum()
    # Get the total time of the search flight:
    total_time_search = bdf[bdf["flight_type"] == "search"]["t"].iloc[-1] - bdf[bdf["flight_type"] == "search"]["t"].iloc[0]
    # Calculate the average speed of the bee:
    average_speed_search = search_length / total_time_search
    #average_speed_search = bdf[bdf["dt"] < 4]["dv"].mean()
    # Calculate the distance from the respective release site:


    print(bee)
    release_site = bdf["release_site"].iloc[0]
    print(release_site)

    # Create a new dataframe for the bee with columns bee, release_site, vector, search:
    df_temp = pd.DataFrame({"Bee": [bee], 
                            "release_site": [release_site], 
                            "search_length": [search_length],
                            "average_speed_search": [average_speed_search],
                            "total_time_search": [total_time_search]})                            
                        
    # Concatinate the dataframe to df2:
    df_output = pd.concat([df_output, df_temp], axis=0, ignore_index=True)

# Rename "Hive_release" to "HR":
df_output["release_site"] = df_output["release_site"].replace("Hive_release", "HR")
df["release_site"] = df["release_site"].replace("Hive_release", "HR")

# Set the figure labels
plot_title = "Search flight characteristics"
subplot_labels = ["A", "B", "C", "D"] #, "E"]
subplot_titles = ["Distance to\nF1r or F1v",
                    "Distance to\nrespective release site",
                    "Flight length",
                  #  "Distance to feeder F1r",
                    "Average speed"]
ylabel_list = ["Distance (m)", "Distance (m)", "Length (m)", "Speed (m/s)"]

# Rename df to df_search:
df_search = df
del df

dataframe_list = [df_search, df_search, df_output, df_output]
parameter_list = ["distance_to_virtual_feeder",
                  "distance_from_release_site",
                  "search_length",
                 # "distance_to_feeder",
                  "average_speed_search"]


# Set style to scale font and line width:
fig, ax = plt.subplots(1, 4, figsize=(12, 5))
#sns.set(font_scale=1.5, style="ticks")
# Set the figure scale factor:
sns.set(font_scale=1.2, style="ticks")

# Define palette colors:
palette = ["white", "lightgray", "darkgray"]

for i, (df, parameter) in enumerate(zip(dataframe_list, parameter_list)):
    # Create a boxplot for each dataframe and parameter:
    sns.boxplot(ax=ax[i], x="release_site", y=parameter, data=df, order=["HR", "R1", "R2"], palette=palette)
    if i > 1:
        sns.swarmplot(ax=ax[i], x="release_site", y=parameter, data=df, color=".25", order=["HR", "R1", "R2"], size=4)

    ax[i].set_xlabel("")
    ax[i].set_title(subplot_titles[i], fontsize=15, y=1.05)
    ax[i].set_ylabel(ylabel_list[i], fontsize=14)

    # Set x and y tick size:
    ax[i].tick_params(axis='x', labelsize=12)
    ax[i].tick_params(axis='y', labelsize=12)

    # Set subplot label letter:
    ax[i].text(-0.1, 1.2, subplot_labels[i], transform=ax[i].transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')
    
    _, res = add_stat_annotation(ax=ax[i], data=df, x="release_site", y=parameter,
                    order=["HR", "R1", "R2"], box_pairs=[("HR", "R1"), ("HR", "R2"), ("R1", "R2")],
                    test='Kruskal', text_format=p_value_style, loc='inside')
    
    res

    for r in res:
                # Store them in a dataframe:
        df_temp = pd.DataFrame({"Subplot of Figure 6:": [subplot_labels[i] + " (" + subplot_titles[i] + ")"],
                                "Box pair": [r.box1 + " vs " + r.box2],
                                "Test name": [r.test_short_name],
                                "Statistic": [r.stat],
                                "p-value": [r.pval]})
        # Concatinate the dataframe to df_stats:
        df_stats = pd.concat([df_stats, df_temp], axis=0, ignore_index=False)
# # Set title:
# plt.suptitle(plot_title, fontsize=16)

#despline the top and right axis:
sns.despine()
# Tight layout:
plt.tight_layout()
# Save the figure:
plt.savefig(figure_outname, dpi=200)

# Save the dataframe to a csv file:
if save_output_to_csv:
    df_output.to_csv(output_csv_name, index=False, header=True)
    # Save the stats table to a csv file:
    df_stats.to_csv(stats_csv_name, index=False, header=True)