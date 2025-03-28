import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import scipy.stats as stats
import numpy as np

# Set the figure labels
plot_title = "Vector flight characteristics"
subplot_labels = ["A", "B", "C", "D"]
subplot_titles = ["Flight length\n(all bees)", 
                  "Flight length\n(only performers)",
                  "Flight speed\n(only performers)",
                  "Straightness\n(only performers)"]
ylabel_list = ["Length (m)", "Length (m)", "Speed (m/s)", "Straightness"]

input_file = "tables/all_data.csv"

figure_outname = "figures/Fig3_kruskal.png"
p_value_style = "star"

# Set the following variables to True if you want to generate new csv files:
save_output_to_csv = True
output_csv_name = "tables/Fig3_vector_flight_characteristics.csv"
save_variance_table = True
variance_csv_name = "tables/Fig3_vector_flight_variance.csv"
stats_csv_name = "tables/Fig3_vector_flight_stats.csv"

# Read the json file with the coordinates of the release sites and the feeder:
with open("landmark_coordinates.json") as file:
    params = json.load(file)
for key, value in params.items():
    globals()[key] = value

# Read input file into a dataframe:
df = pd.read_csv(input_file)

# Create an empty dataframe to store the flight characteristics:
df_output = pd.DataFrame()

# Iterate through bees and calculate the vector flight characteristics:
for bee, bdf in df.groupby("Bee"):
    # Check if release site is Hive_Release. If so, rename to HR:
    bdf["release_site"] = bdf["release_site"].replace("Hive_release", "HR")

    # Check if the bee has performed a vector flight and calculate the vector flight characteristics:
    if "vector" in bdf["flight_type"].unique():
        print("Vector flight performed", bee)
        #calculate the total distance of when flight_type is vector:
        vector_length = bdf[bdf["flight_type"] == "vector"]["distance_from_previous_position"].sum()
        # Get the total time of the vector flight:
        total_time_vector = bdf[bdf["flight_type"] == "vector"]["t"].iloc[-1] - bdf[bdf["flight_type"] == "vector"]["t"].iloc[0]
        # Calculate the average speed of the bee:
        average_speed_vector = vector_length / total_time_vector
        # Take the last distance_from_release_site valuefor vector flight:
        distance_from_release_site_straight = bdf[bdf["flight_type"] == "vector"]["distance_from_release_site"].iloc[-1]
        straightness = distance_from_release_site_straight / vector_length
    else:
        # Set everything to zero if the bee has not performed a vector flight:
        vector_length = 0
        average_speed_vector = 0
        distance_from_release_site_straight = 0
        straightness = 0
        print("No vector flight for bee: ", bee)
    release_site = bdf["release_site"].iloc[0]
    # Create a new dataframe for the bee with columns bee, release_site, vector, search:
    df_temp = pd.DataFrame({"Bee": [bee], 
                            "release_site": [release_site], 
                            "vector_length": [vector_length], 
                            "vector_straight_line": [distance_from_release_site_straight],
                            "straightness": [straightness],
                            "average_speed_vector": [average_speed_vector], 
                           })    
    # Concatinate the dataframe to df_output:
    df_output = pd.concat([df_output, df_temp], axis=0, ignore_index=True)

# Save the dataframe to csv:
if save_output_to_csv:
    df_output.to_csv(output_csv_name, index=False, header=True)

# create a new dataframe that excludes bees with zero vector_length:
df_performers_only = df_output[df_output["vector_length"] != 0]

# Plot three boxplots in one figure with six subplots arranged in two rows and three columns:
plt.figure(figsize=(12, 5))

# Generate the figure by running a loop with these parameters:
dataframe_list = [df_output, df_performers_only, df_performers_only, df_performers_only]
parameter_list = ["vector_length", "vector_length", "average_speed_vector", "straightness"]

# Set style to scale font and line width:
sns.set(font_scale=1.2, style="ticks")

# Define palette colors:
palette = ["white", "lightgray", "darkgray"]

# Create an empty dataframe to store the variance and levene test results:
df_var = pd.DataFrame()
df_stats = pd.DataFrame()

# Set to True to ignore FutureWarnings:
if True:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

for dataframe, parameter, subplot, ylabel, subplot_label, subplot_title in zip(dataframe_list, parameter_list, [1,2,3,4], ylabel_list, subplot_labels, subplot_titles):
    # Print figure letter and parameter:
    print("\n\nFigure 3.", subplot_label, parameter)

    # Increase the subplot number by 1:
    plt.subplot(1, 4, subplot)
    ax = sns.boxplot(x="release_site", y=parameter, data=dataframe, order=["HR", "R1", "R2"], palette=palette)
    ax = sns.swarmplot(x="release_site", y=parameter, data=dataframe, color=".25", order=["HR", "R1", "R2"], size=4)

    ax.set_xlabel("")
    ax.set_title(subplot_title, fontsize=15, y=1.05)
    ax.set_ylabel(ylabel, fontsize=14)
    # Set x and y tick size:
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Set subplot label letter:
    ax.text(-0.1, 1.2, subplot_label, transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')
   

   # plt.ylabel(ylabel)
   # plt.xlabel("")
   # # Set subplot label letter:
   # plt.text(-0.2, 1.3, subplot_label, transform=plt.gca().transAxes,
   #     fontsize=16, fontweight='bold', va='top', ha='right')
   # # Set the title of the subplot:
   # plt.title(subplot_title, fontsize=15, y=1.1)

    # Annote the plot:
    _, res = add_stat_annotation(ax=plt.gca(), data=dataframe, x="release_site", y=parameter,
                    order=["HR", "R1", "R2"],
                    box_pairs=[("HR", "R1"), ("HR", "R2"), ("R1", "R2")],
                    test='Kruskal', text_format=p_value_style, loc='inside',
                    )
    res

    for r in res:

        # Store them in a dataframe:
        df_temp = pd.DataFrame({"Subplot of Figure 3:": [subplot_label + " (" + subplot_title + ")"],
                                "Box pair": [r.box1 + " vs " + r.box2],
                                "Test name": [r.test_short_name],
                                "Statistic": [r.stat],
                                "p-value": [r.pval]})
        # Concatinate the dataframe to df_stats:
        df_stats = pd.concat([df_stats, df_temp], axis=0, ignore_index=False)


    if save_variance_table:
        # Compare the variance between the three groups:
        var_HR = dataframe[dataframe["release_site"] == "HR"][parameter].var()
        var_R1 = dataframe[dataframe["release_site"] == "R1"][parameter].var()
        var_R2 = dataframe[dataframe["release_site"] == "R2"][parameter].var()
        levene_HR_R1 = stats.levene(dataframe[dataframe["release_site"] == "HR"][parameter], dataframe[dataframe["release_site"] == "R1"][parameter])
        levene_HR_R2 = stats.levene(dataframe[dataframe["release_site"] == "HR"][parameter], dataframe[dataframe["release_site"] == "R2"][parameter])
        levene_R1_R2 = stats.levene(dataframe[dataframe["release_site"] == "R1"][parameter], dataframe[dataframe["release_site"] == "R2"][parameter])
        
        # Concatinate the variances and levene test results into a dataframe with the 
        # column names "subplot_label", "parameter", "variance" and "Levene test result" and
        # "Levene test p-value":
        df_temp = pd.DataFrame({"Subplot of Figure 3:": [subplot_label], "variance_HR": [var_HR], "variance_R1": [var_R1], "variance_R2": [var_R2],
                                "Levene test HR vs R1": [levene_HR_R1.statistic], "Levene test HR vs R1, p-value": [levene_HR_R1.pvalue],
                                    "Levene test HR vs R2": [levene_HR_R2.statistic], "Levene test HR vs R2, p-value": [levene_HR_R2.pvalue],
                                    "Levene test R1 vs R2": [levene_R1_R2.statistic], "Levene test R1 vs R2, p-value": [levene_R1_R2.pvalue]}) 
        df_var = pd.concat([df_var, df_temp], axis=0, ignore_index=False)

if save_variance_table:
    # Save the variance dataframe to csv:
    df_var.to_csv(variance_csv_name, index=False, header=True)

# Save the stats dataframe to csv:
if save_output_to_csv:
    df_stats.to_csv(stats_csv_name, index=False, header=True)

# # Set title:
# plt.suptitle(plot_title, fontsize=16)

#despline the top and right axis:
sns.despine()
# Tight layout:
plt.tight_layout()

# Save the figure:
plt.savefig(figure_outname, dpi=200)

# print length od df_output and df_performers_only:
print("Length of df_output: ", len(df_output), "Length of df_performers_only: ", len(df_performers_only))