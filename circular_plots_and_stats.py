# This script imports csv data from all_data.csv and plots it in a polar histogram with 0 degree
# pointing north (top) and 90 degrees pointing east (right). The data is in degrees, so it is
# converted to radians for the polar plot. 
# This script reads in a csv file containing the counts of bee headings for each release site,
# performs a test for circular uniformity, and calculates the circular mean.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pycircstat
from scipy.stats import circstd

# Set number of bins
N = 72

# Read data from csv file
df_all = pd.read_csv('tables/all_data.csv')

flight_types = ['vector','search']

for flight_type in flight_types:

       #Only use one type flights
       df = df_all[df_all['flight_type'] == flight_type]

       # Create empty dataframe to collect thetas and counts into:
       df_hist = pd.DataFrame()

       # Create empty dataframe to collect statistics into:
       df_stats = pd.DataFrame()

       # Make three plots for different release sites
       for release_site, rdf in df.groupby('release_site'):

              # For alpha1, only use data where dt is less than 4 second and store it in a new dataframe:
              rdf1 = rdf[rdf['dt'] < 4].reset_index(drop=True)

              # Extract data from dataframe
              alpha1 = rdf1['bearing_from_previous_position']
              alpha2 = rdf['bearing_from_release_site']

              # Compute histogram with numpy
              counts1, theta1 = np.histogram(alpha1, bins=N, range = [0,360])
              counts2, theta2 = np.histogram(alpha2, bins=N, range = [0,360])

              # Compute the center of each bin (theta) and convert to radians
              theta1 = (theta1[:-1]+theta1[1:])/2
              theta1 = np.deg2rad(theta1)
              theta2 = (theta2[:-1]+theta2[1:])/2
              theta2 = np.deg2rad(theta2)

              # Store release site and all data in a dataframe
              df_temp = pd.DataFrame()
              df_temp['theta1'] = theta1
              df_temp['counts1'] = counts1
              df_temp['theta2'] = theta2
              df_temp['counts2'] = counts2
              df_temp['release_site'] = release_site

              colors1 = plt.cm.Blues(counts1 / np.max(counts1))
              width1 = 2 * np.pi / N * np.ones_like(theta1)
              colors2 = plt.cm.Blues(counts2 / np.max(counts2))
              width2 = 2 * np.pi / N * np.ones_like(theta2)

              # Concatinate the dataframe to alpha vertically:
              df_hist = pd.concat([df_hist, df_temp], axis=0)

              # Perform test for circular uniformity for each release site:
              angles1 = alpha1.apply(np.deg2rad)
              angles2 = alpha2.apply(np.deg2rad)

              # Drop nan in angles1:
              angles1 = angles1.dropna()
              r1 = np.sqrt(np.sum(np.cos(angles1))**2 + np.sum(np.sin(angles1))**2) / len(angles1)
              print("Mean resultant vector of alpa1", r1)
              r2 = np.sqrt(np.sum(np.cos(angles2))**2 + np.sum(np.sin(angles2))**2) / len(angles2)
              print("Mean resultant vector of alpa2", r2)

              # Calculate circular mean and circular standard deviation:
              mean1 = np.rad2deg(np.arctan2(np.sum(np.sin(angles1)), np.sum(np.cos(angles1))))
              mean2 = np.rad2deg(np.arctan2(np.sum(np.sin(angles2)), np.sum(np.cos(angles2))))

              # Convert radians to degrees
              if mean1 < 0:
                     mean1 = mean1 + 360
              print("Circular mean of alpha1", mean1)

              # Convert radians to degrees
              if mean2 < 0:
                     mean2 = mean2 + 360
              print("Circular mean of alpha2", mean2)

              # Calculate circular standard deviation:
              std1 = np.rad2deg(circstd(angles1))
              print("Circular standard deviation of alpha1", std1)
              std2 = np.rad2deg(circstd(angles2))
              print("Circular standard deviation of alpha2", std2)

              # Perform Rayleigh test for circular uniformity:
              z1,p1 = pycircstat.tests.rayleigh(angles1)
              print("Rayleigh test of alpha1, z = " , z1 , ", p = " , p1)
              z2,p2 = pycircstat.tests.rayleigh(angles2)
              print("Rayleigh test of alpha2,z = " , z2 , ", p = " , p2)

              # Create dataframe to collect statistics into:
              df_temp2 = pd.DataFrame({'release_site': [release_site],
                            'Alpha 1': [''],          
                            'Mean resultant vector of alpha1': [r1],
                            'Circular mean of alpha1': [mean1],
                            'Circular standard deviation of alpha1': [std1],
                            'Rayleigh test of alpha1 (z)': [z1],
                            'Rayleigh test of alpha1 (p)': [p1],
                            'Alpha 2': [''],
                            'Mean resultant vector of alpha2': [r2],
                            'Circular mean of alpha2': [mean2],
                            'Circular standard deviation of alpha2': [std2],
                            'Rayleigh test of alpha2 (z)': [z2],
                            'Rayleigh test of alpha2 (p)': [p2]
                             })
              df_stats = pd.concat([df_stats, df_temp2], axis=0)

       # Save df_hist as csv file
       df_hist.to_csv("tables/circular_histogram_values_" + flight_type + "_" + str(N) + "_bins.csv", index=False)

       # Rotate df_stats by 90 degrees:
       df_stats = df_stats.T

       # Save df_stats as csv file
       df_stats.to_csv("tables/circular_statistics_" + flight_type + ".csv", index=True, header=0)

       # Run a for loop to simplify code:
       count_types = ["counts1", "counts2"]
       theta_types = ["theta1", "theta2"]
       subplot_labels = ["A", "D", "B", "E", "C", "F"]
       release_sites = ["HR", "R1", "R2"]

       # Create a list of numbers from 1 to 6:
       subplot_numbers = [1,4,2,5,3,6]

       i = 0

       # Get ymax value for y-axis in polar plot from the maximum value in the dataframe:
       ylim_max = df_hist[count_types].max().max()
       print(ylim_max)

       # Create a figure with polar plots, 2 rows, 3 columns:
       fig, axs = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(10,6))

       for release_site, rdf in df_hist.groupby('release_site'):
              first_iteration = True
              for count_type, theta_type in zip(count_types, theta_types):
                     subplot_label = subplot_labels[i]
                     subplot_number = subplot_numbers[i]

                     print(release_site, count_type, theta_type)

                     #colors="lightgrey"
                     colors=plt.cm.viridis(rdf[count_type] / np.max(rdf[count_type]))
                     width = 2 * np.pi / N * np.ones_like(rdf[theta_type])

                     plt.subplot(2, 3, subplot_number)
                     plt.bar(rdf[theta_type], rdf[count_type], width=width, bottom=0, color=colors, alpha=0.75,
                            linewidth=1.5, edgecolor='k')
                     
                     # Set zero location to North and direction to clockwise
                     plt.gca().set_theta_zero_location("N")
                     plt.gca().set_theta_direction(-1)
                     # Rotate y-labels:
                     plt.gca().set_rlabel_position(135)

                     # Set xticks to cardinal directions (set to True to turn on)
                     if False:
                            plt.gca().set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

                     # Make grid translucent and yticks grey:
                     plt.gca().xaxis.grid(alpha=.25)
                     plt.gca().yaxis.grid(alpha=.25)
                     plt.gca().tick_params(axis='y', colors='grey')
                     plt.text(-.1, 1.1, subplot_label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
                     # Set ylim:
                     plt.ylim(0, ylim_max+10)
                     # Set subplot title to release site:
                     if first_iteration:
                            plt.text(0.5, 1.3, release_site, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
                            first_iteration = False  

                     i += 1
                     # despline the plot
                     #plt.gca().spines['polar'].set_visible(False)

       alpha_charac = r'$\alpha$'

       # Add a y-label to the leftmost column:
       plt.subplot(2, 3, 1)
       plt.text(-.4, 0.5, alpha_charac + " 1", transform=plt.gca().transAxes, fontsize=16, va='center', rotation='vertical')

       # Add a y-label to the leftmost column:
       plt.subplot(2, 3, 4)
       plt.text(-.4, 0.5, alpha_charac + " 2", transform=plt.gca().transAxes, fontsize=16, va='center', rotation='vertical')

       # Tight layout
       plt.tight_layout()

       # Save figure as png
       fig_outname = 'figures/Fig_' + flight_type + '_polar_plot_' + str(N) + '_bins.png'
       plt.savefig(fig_outname, dpi=200)
       plt.close()