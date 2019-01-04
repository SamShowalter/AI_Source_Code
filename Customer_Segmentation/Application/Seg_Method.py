###########################################################################
#
# Segmentation Method Interface 
# 
#
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

# #Scientific computing libraries
import pandas as pd


#Package specific imports
from Seg_Viz import *
from Logger import Log


###########################################################################
# Class and constructor
###########################################################################

class SegMethod():
	"""
	This is a de-facto interface for all of the potential segmentation methods. Python does not have an explicit
	interface class so this method will be used instead. The methods outlined here will be implemented by all sub-classes

	Attributes:
				### Information pertaining to class object interation ###

			CustomerSeg: 				Customer Seg test object
			Method:						Method implementation class (e.g. KMeans)
			MethodLog:					Log object for Segmentation Method execution

				### Information pertaining to class execution ###

					-- Elbow plot analysis variables --

			cluster_results:			Results from cluster analysis. A Dataframe
			cluster_range_start:		Smallest number of clusters to analyze
			cluster_range_end:			Largest number of clusters to analyze
			elbow_plot_viz_filename:	Elbow plot visualization filename
			elbow_subtest_cluster_num:	Number of clusters for an elbow subtest
			elbow_subtest_sqr_dist:		Squared distance for elbow subtest
			is_sub_action:				Boolean that identifies if action is a 
										sub-routine of a larger process.

					-- Single Run analysis varianbles --  

			single_cluster_num:			Cluster number for single run analysis
			single_sqr_dist:			Cluster squared distance for single run analysis


				### Information pertaining to class metadata and logging ###

			execution_date_start:		Start date of execution. Changed for each new action
			execution_time_start:		Start date of execution. Changed for each new action 
			seg_method_name:			Name of the clustering method used
			MethodLogFilename:			File name of the Method log created.
			action_dict:				Dictionary managing all actions taken 
										by object for master log
			num_method_actions:			Number of actions taken by SegMethod object

	"""

	def __init__(self, 
				 CustomerSeg,
				 Method):
		"""
		Constructor for the Segmentation Method class. This primarily assigns and initializes the actual 
		implementation method that will be used. 

		Args:

				### Information pertaining to class object interation ###

			CustomerSeg:				Customer Segmentation test object
			Method:						SegMethod implementation object (e.g. KMeans)

		"""

		#Attribute class related object and names to class
		self.CustomerSeg = CustomerSeg
		self.Method = Method
		self.seg_method_name = self.Method.name

		#Instantiate preprocess log
		self.MethodLog = Log("Not Applicable", "Not Applicable", "%s_Method-Log"%(CustomerSeg.test_name), directory = self.CustomerSeg.Log.directory)
		self.MethodLog.master_log_name = self.CustomerSeg.Log.master_log_name
		self.MethodLog.test_number = self.CustomerSeg.Log.test_number

		#Set method log filename
		self.MethodLogFilename = self.MethodLog.method_log_name + ".csv"

		#Action dictionary
		self.action_dict = {}

		#Start execution duration seconds
		self.execution_duration_sec = 0

		#Number of seg_method actions
		self.num_method_actions = 0

	###########################################################################
	# Public Methods for Unsupervised Learning -- User-called
	###########################################################################

	def elbow_chart_test(self, 
						 cluster_range, 
						 viz = False, 
						 viz_name = "",
						 show = False,
						 profile = False):
		"""
		The elbow chart test seeks to determine the optimal number
		of clusters to use to segment data. It does so by determining 
		the mean squared distance of data from centroids. This test relies
		on a visualization that shows where the "elbow" of the plot is, or 
		where the slope of the line decreases the most. This method calls
		its corresponding private method inside the decorator function handler.

		Args:

			cluster_range:					Number range of clusters to examine
			viz:							Boolean determining if a graph should be made
											(For this function, it always should be made)
			viz_name:						Name of the visualization
			show:							Boolean on whether or not to show visual at 
											runtime. Otherwise its just saved.							
		"""
		
		#Function handler call for metadata
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private function
													 self.__elbow_chart_test,

													 #All local variable inputs
													 locals(), 

													 #Variables to de-allocate
													 ["elbow_cluster_range_start",
													  "elbow_cluster_range_end",
													  "elbow_plot_viz_filename"])

	

	def cluster(self, 
				num_clusters,
				profile = False):
		"""
		This method runs the clustering algorithm for its given
		implementation, and does so for a single parameter set.
		This method calls its corresponding private method inside 
		the decorator function handler.

		Args:

			num_clusters:				Number of clusters for segmenting data

		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private function
													 self.__cluster,

													 #All local variables
													 locals(),

													 #Variables to de-allocate
													 ["single_cluster_num",
													  "single_sqr_dist"])

	###########################################################################
	# Private Methods for Unsupervised Learning -- Implementation
	###########################################################################

	def __elbow_chart_test(	self, 
							cluster_range = range(0,0), 
							viz = False, 
							viz_name = "",
							show = False,
							profile = False):
		"""
		Private method that runs the implementation code for 
		elbow chart test. Its default keyword arguments are
		specifically set to cause errors if ever called, 
		though all of these kwargs will be filled by its 
		user-run function. It is set that way to ensure
		users are accessing these functions correctly.

		Args:

			cluster_range:					Number range of clusters to examine
			viz:							Boolean determining if a graph should be made
											(For this function, it always should be made)
			viz_name:						Name of the visualization
			show:							Boolean on whether or not to show visual at 
											runtime. Otherwise its just saved.			
		"""

		#Initialize cluster results and other system variables
		self.cluster_results = []
		self.elbow_cluster_range_start = cluster_range[0]
		self.elbow_cluster_range_end = cluster_range[-1]

		#Iterate through each cluster number in the specified range
		for cluster_num in cluster_range:

			#Run the elbow chart sub-routine -- implements analysis
			self.__elbow_chart_sub_routine(cluster_num,
										   viz,
										   viz_name,
										   show)

			#If the profiling is requested
			if profile:

				#Generate and save a data profile
				self.CustomerSeg.Preprocess.generate_profile(self.cluster_results[-1][1], 
															 cluster_num,
															 save = True)

		#Transform the results to be a dataframe
		self.cluster_results = pd.DataFrame(self.cluster_results)


		if viz:
			#Create visualization
			elbow_plot_pic = elbow_plot(cluster_range, self.cluster_results[2], viz_name)

			#Show visualization at runtime
			if show:
				elbow_plot_pic.show()

			#Save image and log name metadata
			self.CustomerSeg.Log.saveImage(elbow_plot_pic, "elbow_plot", self.CustomerSeg.viz_folder_name)
			self.elbow_plot_viz_filename = self.CustomerSeg.Log.image_name

		#Action name metadata
		self.action_name = "elbow_chart_test"


	def __elbow_chart_sub_routine( self, 
								 cluster_num, 
								 viz = False, 
								 viz_name = "",
								 show = False,
								 profile = False):
		"""
		This sub-routine is a scheduling function so that sub-routine metadata can be stored.

		Args:

			cluster_num:					Number of clusters to examine
			viz:							Boolean determining if a graph should be made
											(For this function, it always should be made)
			viz_name:						Name of the visualization
			show:							Boolean on whether or not to show visual at 
											runtime. Otherwise its just saved.	
		"""
		
		locals()['self'].CustomerSeg.functionHandler(self, 

													 # Corresponding private function
													 self.__elbow_chart_sub_routine_helper,

													 # All local variables
													 locals(), 

													 # Variables to de-allocate
													 ["elbow_subtest_cluster_num",
													  "elbow_subtest_sqr_dist",
													  "is_sub_action"])


	def __elbow_chart_sub_routine_helper(self,
										 cluster_num = 0,
										 viz = False,
										 viz_name = "",
										 show = False,
										 profile = False):
		"""
		This sub-routine helper implements the elbow chart sub routine code.
		Primarily, this code clusters data based on a specific number of 
		clusters.

		Args:

			cluster_num:					Number of clusters to examine
			viz:							Boolean determining if a graph should be made
											(For this function, it always should be made)
			viz_name:						Name of the visualization
			show:							Boolean on whether or not to show visual at 
											runtime. Otherwise its just saved.	
		"""

		#Metadata variable assignment
		self.action_name = "elbow_chart_sub_routine"
		self.is_sub_action = True
		self.elbow_subtest_cluster_num = cluster_num 

		#Results from cluster implementation run
		self.elbow_clust_result = self.Method.cluster(self.CustomerSeg.train_data,
												 cluster_num,
												 self.CustomerSeg.random_state)

		#Add those results to the cluster results dictionary
		self.cluster_results.append(self.elbow_clust_result)
		self.elbow_subtest_sqr_dist = self.elbow_clust_result[2]


	def __cluster(self, 
				  num_clusters = 0,
				  profile = False):
		"""
		General cluster function that will implement in different ways 
		depending on the underlying solver. In general, though, it applies to
		K-means clustering.

		Args:

			num_clusters:		Number of clusters.
		
		Implementations:

			K_Means:			K_means clustering done through sklearn
		"""

		#Define the cluster range start and end
		self.cluster_range_start = cluster_range[0]
		self.cluster_range_end = cluster_range[-1]

		#Save the cluster results for single clust(er)
		#This name is funky so it doesn't overwrite another variable
		self.clust_results = self.Method.cluster(self.CustomerSeg.train_data,
												 num_clusters,
												 self.CustomerSeg.random_state)

		#If the profiling is requested
		if profile:

			#Generate and save a data profile
			self.CustomerSeg.Preprocess.generate_profile(self.clust_results[1], 
														 num_clusters,
														 save = True)

		#Name metadata
		self.action_name = "single_cluster"





	###########################################################################
	# Public Methods for Supervised Learning
	###########################################################################



	###########################################################################
	# Private Helper Methods
	###########################################################################

	