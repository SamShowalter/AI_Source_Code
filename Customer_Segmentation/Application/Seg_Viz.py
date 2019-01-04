###########################################################################
#
# Segmentation Visualization File (Static)
# 
#
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Scientific Computing Libraries
import numpy as np

#Visualization imports
import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Change matplotlib style to look like R (for Jamie :) )
mpl.style.use('ggplot') 
import seaborn as sns 

#Package specific imports
# None yet

###########################################################################
# Public Methods
###########################################################################

"""

	This is a static class for visualizations that will be leveraged throughout the execution
	process. It makes more sense to house all of these together as many do not explicitly belong 
	to one class object. Any and all of them can be referenced here.

"""

##############################
# PCA - based plots
##############################


def cumulative_variance(data, 
						viz_name = ""):
	"""
	Cumulative variance plot, corresponds to Principal Component Analysis
	process. This looks at how much of the percentage cumulative variance 
	is explained by different numbers of PCA components.

	Args:

		data:				pca cumulative variance data
		viz_name:			Name for the visualization

	Returns:

		matplotlib.pyplot object with figure

	"""

	#Initialize a new figure for the plot 
	plt.figure()

	#Plot the number of components and the cumulative variance
	plt.plot(range(1, len(np.cumsum(data)) + 1), np.cumsum(data))

	#Set X and Y axis names
	plt.xlabel('Number of Components')
	plt.ylabel('Cumulative Explained Variance')

	#Give the visualization a name if necessary
	if viz_name != "":
		viz_name = ": " + viz_name

	#Set the name of the title
	plt.title('Cumulative Variance' + viz_name)

	#Return the plot
	return plt


##############################
# K-means
##############################

def elbow_plot(clusters_attempted, 
			   cluster_distance_res, 
			   viz_name = ""):
	"""
	Elbow plots pertain to iteratively determining which number of clusters
	is optimal for clustering a dataset. The metric of success is minimizing
	the sum of squared distance of data points from the centroids

	Args:

		clusters_attempted:			Range of clusters that were tried
		cluster_distance_res:		Distance results from each k-means analysis
		viz_name:					Visualization name

	Returns:

		matplotlib.pyplot object with figure

	"""

	#Initialize the figure
	plt.figure()

	#Plot the clusters attempted with their distance
	plt.plot(clusters_attempted,cluster_distance_res)

	#Label the X and Y axes
	plt.xlabel('Number of Clusters')
	plt.ylabel('Sum of Squared Distance')

	#Add a visualization name if relevant
	if viz_name != "":
		viz_name = ": " + viz_name

	#Add the title
	plt.title('Elbow Plot' + viz_name)

	#return the plot
	return plt

#####################################
# K-means -- Silhouette Analysis Viz
#####################################

def silhouette_viz(CustomerSeg, 
				   cluster_labels, 
				   cluster_num, 
				   silhouette_avg,
				   sample_silhouette_values,
				   x_feature_index, 
				   y_feature_index,
				   x_feature_name,
				   y_feature_name,
				   viz_name = ""):

		"""
		Visualization of silhouette analysis. This takes code from 
		a sci-kit learn example and modifies it to better work with 
		the function. The source is cited below.

		Args:

				### Object Specific Imports ###

			CustomerSeg:						Customer Segmentation test object
			cluster_labels;						Labels for the clusters
			cluster_num:						Number of clusters
			silhouette_avg:						Average SilhouetteScore
			sample_silhouette_values:			silhouette scores for each data point
			x_feature_index:					Feature to be displayed on X-axis
			y_feature_index:					Feature to be displayed on Y-axis
			x_feature_name:						Name for X-axis
			y_feature_name:						Name for Y-axis
			viz_name:							Visualization name

		Returns:

			matplotlib.pyplot object with figure

		Code_Source:

			http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

		"""

		# Create a subplot with 1 row and 2 columns
		#Set the size of the plot
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from [-1, 1] 
		# However, typically the scores will be positive
		ax1.set_xlim([-0.2, 1])


		# The (cluster_num+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(CustomerSeg.train_data) + (cluster_num + 1) * 10])
		y_lower = 10

		#For each cluster number
		for i in range(cluster_num):

		    # Aggregate the silhouette scores for samples belonging to
		    # cluster i, and sort them
		    ith_cluster_silhouette_values = \
		        sample_silhouette_values[cluster_labels == i]
		    ith_cluster_silhouette_values.sort()

		    # Set the size of the cluster, and adjust viz size
		    size_cluster_i = ith_cluster_silhouette_values.shape[0]
		    y_upper = y_lower + size_cluster_i

		    # Set colors and fill in the distances shape for
		    # each data point silhouette score
		    color = cm.nipy_spectral(float(i) / cluster_num)
		    ax1.fill_betweenx(np.arange(y_lower, y_upper),
		                      0, ith_cluster_silhouette_values,
		                      facecolor=color, edgecolor=color, alpha=0.7)

		    # Label the silhouette plots with their cluster numbers at the middle
		    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		    # Compute the new y_lower for next plot
		    y_lower = y_upper + 10  # 10 for the 0 samples

		#Set the title and axis labels
		ax1.set_title("The silhouette plot for the various clusters: " + viz_name)
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# 2nd Plot showing the actual clusters formed
		colors = cm.nipy_spectral(cluster_labels.astype(float) / cluster_num)

		ax2.scatter(CustomerSeg.train_data.iloc[:, 0], 
					CustomerSeg.train_data.iloc[:, 1], 
					marker='.', s=30, lw=0, alpha=0.7,
		            c=colors, edgecolor='k')

		# Labeling the clusters
		centers = CustomerSeg.SegMethod.cluster_results.iloc[cluster_num - 2, 0]

		# Draw white circles at cluster centers
		ax2.scatter(centers[:, x_feature_index], centers[:, y_feature_index], marker='o',
		            c="white", alpha=1, s=200, edgecolor='k')

		#Plot the clusters with the centroids
		for i, c in enumerate(centers):
		    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
		                s=50, edgecolor='k')

		#Set axis plot and labels
		ax2.set_title("The visualization of the clustered data:" + viz_name)
		ax2.set_xlabel(x_feature_name)
		ax2.set_ylabel(y_feature_name)

		#Set overall title for both plots
		plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
		              "with n_clusters = %d" % cluster_num),
		             fontsize=14, fontweight='bold')

		#Return the plot
		return plt
		
		

################################################
# Scatterplot of clusters (actual or simulated)
################################################

def cluster_viz(train_data, 
				class_label, 
				x_feature_index = 0, 
				y_feature_index = 1,
				hue_index = 2,
				show = False,
				viz_name = ""):
	"""
	The cluster visualization will show the data points on a 2D plane
	based on 2 chosen features. It will also color them based on their
	class label for different clusters. At the moment this is the
	ACTUAL cluster label, but a predicted set of labels could
	also be given as an import. This function may be changed later
	to accommodate that.

	Args:

		train_data:				Train feature data for clustering
		class_label:			Class labels for the feature data
		x_feature_index:		Column index in the train_data Pandas df used for x-axis viz
		y_feature_index:		Column index in the train_data Pandas df used for y-axis viz
		hue_index:				Data index column in the train_data Pandas df that colors
								the scatter points
		show:					Boolean on if you want to see the graph at runtime
		viz_name:				Visualization name

	Returns:

			matplotlib.pyplot object with figure

	"""

	#Create the figure
	plt.figure()

	#Set the class label for the training data
	train_data["class_label"] = class_label

	#Rename the training data X- and Y-features
	train_data.rename(index=str, 
					  columns={x_feature_index: "X_Feature", y_feature_index: "Y_Feature"}, 
					  inplace = True)

	#Create scatterplot of data and color by class labels
	sns.scatterplot(x = "X_Feature", 
					y = "Y_Feature", 
					hue = "class_label",
					data = train_data,

					#Set colors to be easy to understand
					palette = sns.color_palette("Set1", len(set(list(train_data["class_label"])))))

	# Set cluster visualization: May not always be actual but for 
	# now we will keep this name
	plt.title("Actual Cluster Viz: " + viz_name)

	# Show the plot if asked
	if show:
		plt.show()

	#Return the plot
	return plt



###########################################################################
# Private Helper Methods
###########################################################################

	