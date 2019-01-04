###########################################################################
#
# Preprocessing engine for K-means clustering
# (May be made a static file, not sure)
#
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import copy

#Scientific computing libraries
import numpy as np 
import pandas as pd

# #Data Science and predictive libraries
import sklearn as skl 
from sklearn.metrics import silhouette_samples, silhouette_score

#Package specific imports
from Seg_Viz import *
from Logger import Log


###########################################################################
# Class and constructor
###########################################################################

class Preprocess():
	"""
	Preprocessing library for all data analysis steps to prepare data for segmentation
	analysis. 

	Attributes:

				### Information pertaining to class object interation ###

			CustomerSeg: 				Customer Segmentation test object	
			PreprocessLog:				Logger for all preprocess actions		

				### Information pertaining to class execution ###

			action_dict: 				List of actions taken by Preprocessor
			num_preprocess_actions:		Number of preprocess actions taken

					-- Train_test_split variables --

			test_ratio:					ratio of train_data set aside for testing
										(not always used or relevant)
			X_train:					Training feature data
			X_test:						Testing feature data
			y_train:					Training class label
			y_test:						Testing class label

					-- PCA specific variables --

			pca_cum_var_viz_filename:			PCA cumulative variance plot filename
			num_pca_components:					Number of PCA components tested
			cum_var_theshold:					Cumulative variance cut-off threshold
												(chooses the number of PCA components to use)
			explained_var_rat_array:			Explained variance ratio array
			
					-- Silhouette header specific variables -- 

			silhouette_cluster_range_start:		Cluster range start for silhouette analysis
			silhouette_cluster_range_end:		Cluster range end for silhouette analysis
			silhouette_x_feature_index:			X-axis feature for silhouette analysis viz
			silhouette_y_feature_index:			Y-axis feature for silhouette analysis viz

					-- Silhouette header specific variables -- 

			silhouette_viz_filename:			Silhouette visualization file name
			silhouette_subtest_avg_score:		Silhouette average subscore
												(single test represents certain
												 number of clusters)																		  
			silhouette_subtest_cluster_num:		Silhouette analysis subtest cluster number
			is_sub_action:						Is the action a sub-routine of another action


				### Information pertaining to logging and metadata ###

			execution_date_start:				Date of execution start
			execution_time_start:				Time of execution start
			execution_duration_sec:				Total time of preprocess execution
			PreprocessLogFilename:				Name for the preprocess log

	"""

	def __init__(self, CustomerSeg):
		"""
		Constructor for LinearInversion class object. No parameters are given. 
		Necessary information from the estimator class is provided later in the
		execution by QST_sim.

		Args:
			
			CustomerSeg:			Customer Segmentation test object

		"""

		# Attribute all object variables to class
		self.CustomerSeg = CustomerSeg

		#Initialize all metadata and logging variables
		self.action_dict = {}
		self.num_preprocess_actions = 0
		self.execution_duration_sec = 0

		#Instantiate preprocess log
		self.PreprocessLog = Log("Not Applicable", "%s_Preprocess-Log"%(CustomerSeg.test_name), "Not Applicable", directory = 

			self.CustomerSeg.Log.directory)
		self.PreprocessLog.master_log_name = self.CustomerSeg.Log.master_log_name
		self.PreprocessLog.test_number = self.CustomerSeg.Log.test_number

		#Set preprocess log filename
		self.PreprocessLogFilename = self.PreprocessLog.preprocess_log_name + ".csv"


	###########################################################################
	# Public Methods -- will have metadata wrappers
	###########################################################################

	def train_test_split(self, 
						 test_ratio):
		"""
		Train test split user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			test_ratio:					Ratio of train data set aside for testing.
		
		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__train_test_split,

													 #All local variables
													 locals(), 

													 #Vars to de-allocate
													 ["test_ratio"])

	def set_train_data(self, col_names):
		"""
		set training data user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			col_names:				Column names to be set as training data from 
									original data frame.
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_train_data,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_train_data(self, col_indices):
		"""
		set training data user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			col_indices:				Column indives to be set as training data from raw data.
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_train_data,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_train_data(self, 
					   train_data, 
					   col_names = None):
		"""
		Set training data user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			train_data:				Training data for clustering algorithm to use
			col_names:				Names for training data columns

		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_train_data,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_train_col_names(self, 
							col_names):
		"""
		Set training column names user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			col_names:				Column names to label the training data
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_train_col_names,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_class_label(self, 
						col_name):
		"""
		set class label user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			col_name:				Column name to be set as class label
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_class_label,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_class_label(self, 
						col_index):

		"""
		set training data user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			col_index:				Column index to be set as training data
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__set_class_label,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 [])

	def set_class_label(self, 
						column):
		"""
		set class label user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			column:				pd.Series column or list to be made class label
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 # Corresponding private method
													 self.__set_class_label,

													 # All local variables
													 locals(),

													 # Vars to de-allocate
													 [])

	def PCA_fit(self,
				viz = False,
				show = False, 
				viz_name = ""):
		"""
		PCA fit user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			viz:					Boolean that says if you want a visualization made or not	
			show:					Boolean on if you want to see plot at runtime
			viz_name:				Name of the visualization
		"""
		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__PCA_fit,

													 #All local variables
													 locals(),

													 #Vars to de-allocate
													 ['pca_cum_var_viz_filename'])

	def PCA_transform(self,
					  cum_var_threshold = 0.95):
		"""
		PCA transform user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			cum_var_threshold:				PCA cumulative variance threshold to 
											determine number of components
		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__PCA_transform,locals(), 

													 ##Vars to de-allocate
													 ["num_pca_components",
													  "cum_var_threshold",
													  "explained_var_rat_array"])

	def silhouette_analysis(self,
							cluster_range,
							viz = False, 
							viz_name = "",
							show = False,
							x_feature_index = 0, 
							y_feature_index = 1,
							x_feature_name = "PCA First Feature Component",
							y_feature_name = "PCA Second Feature Component"):
		"""
		Silhouette analysis user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			cluster_range:					Range of clusters to test
			viz:							Boolean on whether or not to create visual
			viz_name:						Name tag for the visualization
			show:							Boolean on whether or not to show viz at runtime
			x_feature_index:				Feature index for x-axis on plot
			y_feature_index:				Feature index for y-axis on plot
			x_feature_name:					Name for x-axis on plot
			y_feature_name:					Name for y-axis on plot
		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 # Corresponding private method
													 self.__silhouette_analysis,

													 #All local variables
													 locals(),

													 #Vars to deallocate
													 ['silhouette_cluster_range_start',
													  'silhouette_cluster_range_end',
													  'silhouette_x_feature_index',
													  'silhouette_y_feature_index'])

	def generate_profile(self, 
						 clust_label_preds, 
						 num_clusters, 
						 save = True):
		"""
		Generates a profile for the data, with multiple views saved including
		the mean profile for each feature by cluster, the median profile for each
		feature by cluster, the cluster centers, and the full cluster allocation
		for each dataset

		Args:
			clust_label_preds:			Cluster label predictions
			num_clusters:				Number of clusters
			save:						kwarg on whether or not to save data
		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 # Corresponding private method
													 self.__generate_profile,

													 #All local variables
													 locals(),

													 #Vars to deallocate
													 ["profile_cluster_num",
													  "profile_results_filename"])



	###########################################################################
	# Private Methods -- Sub-routines
	###########################################################################

	def __silhouette_sub_routine(	self,
								cluster_num = 0,
								viz = False, 
								viz_name = "",
								show = False,
								x_feature_index = 0, 
								y_feature_index = 1,
								x_feature_name = "PCA First Feature Component",
								y_feature_name = "PCA Second Feature Component"):
		"""
		Silhouette sub-routine function. This is called by the silhouette
		analysis function to complete more granular logging This function runs its 
		corresponding private method from within the metadata function handler. 
		It also specifies variables that need to be de-allocated after the 
		function has finished.

		Args:

			cluster_num:					Cluster number used in test
			viz:							Boolean on whether or not to create visual
			viz_name:						Name tag for the visualization
			show:							Boolean on whether or not to show viz at runtime
			x_feature_index:				Feature index for x-axis on plot
			y_feature_index:				Feature index for y-axis on plot
			x_feature_name:					Name for x-axis on plot
			y_feature_name:					Name for y-axis on plot
		"""

		locals()['self'].CustomerSeg.functionHandler(self, 

													 #Corresponding private method
													 self.__silhouette_sub_routine_helper, 

													 #All local variables
													 locals(), 

													 #Vars to de-allocate
													 ["silhouette_viz_filename",
													  "silhouette_subtest_avg_score",
													  "silhouette_subtest_cluster_num",
													  "is_sub_action"])


	###########################################################################
	# Private Methods -- Implementation
	###########################################################################

	###################
	# Organizing data
	###################

	def __train_test_split(self, 
						   test_ratio = 0.6):
		"""
		Private train test split implementation function. All arguments are supplied 
		by the user defined function. Determines a random subset of data to be set 
		aside for testing.

		Args:

			test_ratio:					Ratio of train data set aside for testing.

		Raises:

			ValueError:				Makes sure train data and class label vars are set

		"""

		#Set test ratio
		self.test_ratio = test_ratio

		#Name metadata
		self.action_name = "train_test_split"

		#Verify that input data has been formatted appropriately
		if self.CustomerSeg.train_data is None or self.CustomerSeg.class_label is None:
			raise ValueError("Train data or class labels have not yet been set.")

		#Sets all train and test datasets
		self.X_train, self.X_test, self.y_train, self.y_test = tts(self.CustomerSeg.train_data,
																				self.CustomerSeg.class_label,
																				test_ratio,
																				self.CustomerSeg.random_state)



	def __set_train_data(self,
						 col_names = []):
		"""
		
		Private method for setting training data by setting the column names. This
		method slices the input dataframe from the user to make a training dataset.

		Args:

			col_names:				Column names to label the training data

		"""
		#Name metadata
		self.action_name = "set_train_data"

		self.CustomerSeg.train_data = self.data.loc[:,col_names]


	def __set_train_col_names(self,col_names = []):
		"""
		
		Private method for setting training data column name. 

		Args:

			col_names:				Column names to label the training data

		"""
		#Name metadata
		self.action_name = "set_train_col_names"

		self.CustomerSeg.train_data.columns = col_names


	def __set_train_data(self,col_indices = []):
		"""
		
		Private method for setting training data. Slices original 
		dataframe based on indices 

		Args:

			col_indices:				Column indices to label the training data

		"""
		#Name metadata
		self.action_name = "set_train_data"

		self.CustomerSeg.train_data = self.data.iloc[:,col_indices]


	def __set_train_data(self, 
						 train_data = [], 
						 col_names = None):
		"""
		
		Private method for setting training data. Gives the training
		dataframe as an input, as well as optional column names if
		the user wants to set the names for the dataframe as well.

		Args:

			train_data:					Training dataframe
			col_indices:				List of column indices

		"""

		#Name metadata
		self.action_name = "set_train_data"

		#Set the training data
		self.CustomerSeg.train_data = pd.DataFrame(train_data)

		#Set column names if any are given
		if col_names is not None:
			self.CustomerSeg.train_data.columns = col_names


	def __set_class_label(self,
						  col_name = ""):
		"""

		Set class label private method. Determines the class 
		label by slicing original dataframe.
		
		Args:

			col_name:				Column name

		"""
		#Name metadata
		self.action_name = "set_class_label"

		#Set the class label
		self.CustomerSeg.class_label = self.data.loc[:,col_name]


	def __set_class_label(self,
						  col_index = -1):
		"""

		Set class label private method. Determines the class 
		label by slicing original dataframe by index
		
		Args:

			col_index:				Column index to be made class label

		"""

		# Name metadata
		self.action_name = "set_class_label"

		# Set class label
		self.CustomerSeg.class_label = self.data.iloc[:,col_index]


	def __set_class_label(self,column = pd.Series()):
		"""

		Set class label private method. Determines the class 
		label by providing a column of data.
		
		Args:

			column:				Column to be set as class label

		"""

		#Name metadata
		self.action_name = "set_class_label"

		#Set class label
		self.CustomerSeg.class_label = pd.DataFrame(column)


	###################
	# PCA
	###################

	def __PCA_fit(  self, 
					viz = False,
					show = False, 
					viz_name = ""):
		"""
		PCA fit private function. De-dimensionalizes the training data
		and saves it as a PCA object. Also creates a cumulative variance
		plot if requested to determine optimal number of components.

		Args:

			viz:					Boolean that says if you want a visualization made or not	
			show:					Boolean on if you want to see plot at runtime
			viz_name:				Name of the visualization

		Raises:

			ValueError:				Verifies that the train_data has been created

		"""

		#Name metadata
		self.action_name = "pca_fit"

		#Checks to see if training data has been created
		if self.CustomerSeg.train_data is None:
			raise ValueError("Train data has not yet been set.")

		# Initialize and fit data to PCA object
		self.pca = skl.decomposition.PCA()
		self.pca.fit(self.CustomerSeg.train_data)

		#Save variance ratio array
		self.explained_var_rat_array = self.pca.explained_variance_ratio_

		if viz:

			#Create a vizualization of explained variance if requested
			cum_var = cumulative_variance(self.explained_var_rat_array, viz_name)

			# Show plot at runtime, if requested
			if show:
				cum_var.show()

			#Save the image
			self.CustomerSeg.Log.saveImage(cum_var, "cumulative_variance", self.CustomerSeg.viz_folder_name)

			#Save the image filename
			self.pca_cum_var_viz_filename = self.CustomerSeg.Log.image_name


		

	def __PCA_transform(self, 
					    cum_var_threshold = 0.95):
		"""
		PCA transform private method. This method transforms the dataset to represent
		the appropriate amount of PCA components, as determined by the PCA cumulative 
		variance threshold.

		Args:

			cum_var_threshold:				PCA cumulative variance threshold to 
											determine number of components

		Raises:

			ValueError:						Verifies PCA_fit has been executed already

		"""

		#Set cumulative variance threshold for logging
		self.cum_var_threshold = cum_var_threshold

		#Name metadata
		self.action_name = "pca_transform"

		#Make sure that PCA fit has already been run
		if self.action_dict.get("pca_fit") is None:
			raise ValueError("PCA_fit has not been run, therefore PCA component selection cannot yet occur")
		
		#Determine the correct number of PCA components to use
		self.num_pca_components = self.__componentVarTheshold(cum_var_threshold)

		#Set the new training data, store old training data as orig_training_data
		self.CustomerSeg.data = copy.deepcopy(self.CustomerSeg.train_data)
		self.CustomerSeg.train_data = pd.DataFrame(self.pca.transform(self.CustomerSeg.train_data))

		#Save PCA components contribution to each feature
		self.feature_contribution = pd.DataFrame(self.pca.components_,columns=self.CustomerSeg.data.columns)
		#print(self.feature_contribution)

		#Save the pca_composition and log the composition name
		self.CustomerSeg.Log.savePCAComp( self.feature_contribution, "pca_composition")

		#Set the training data so it has the correct number of components
		self.CustomerSeg.train_data = self.CustomerSeg.train_data.iloc[:,range(self.num_pca_components)]


	##############################
	# Silhouette Analysis 
	##############################
	def __silhouette_analysis(self,
							cluster_range = range(0,0),
							viz = False, 
							viz_name = "",
							show = False,
							x_feature_index = 0, 
							y_feature_index = 1,
							x_feature_name = "PCA First Feature Component",
							y_feature_name = "PCA Second Feature Component"):
		"""
		Silhouette analysis private implementation function. This function determines
		the silhouette scores for different cluster analyses, particularly as 
		the number of clusters used to segment the data changes. This function also
		relies on the silhouette sub-routine.

		Args:

			cluster_range:					Range of clusters to test
			viz:							Boolean on whether or not to create visual
			viz_name:						Name tag for the visualization
			show:							Boolean on whether or not to show viz at runtime
			x_feature_index:				Feature index for x-axis on plot
			y_feature_index:				Feature index for y-axis on plot
			x_feature_name:					Name for x-axis on plot
			y_feature_name:					Name for y-axis on plot

		Raises:

			ValueError:						Determines if elbow chart test has been run or not
		"""

		#Set silhouette cluster range start and end vars for logging
		self.silhouette_cluster_range_start = cluster_range[0]
		self.silhouette_cluster_range_end = cluster_range[-1]

		#Set silhouette feature index vars for logging
		self.silhouette_x_feature_index = x_feature_index
		self.silhouette_y_feature_index = y_feature_index

		#Check and see if elbow chart test has been run yet
		if (self.CustomerSeg.SegMethod.action_dict.get("elbow_chart_test") is None):
			raise ValueError("Error: Must run an elbow chart test before conducting silhouette analysis")

		#For each cluster number in the range, run the sub-routine
		for cluster_num in cluster_range:
			self.__silhouette_sub_routine(cluster_num,
										  viz,
										  viz_name,
										  show,
										  x_feature_index,
										  y_feature_index,
										  x_feature_name,
										  y_feature_name)

		#Name metadata
		self.action_name = "silhouette_analysis"
				

	def __silhouette_sub_routine_helper(self, 
									    cluster_num = 0,
										viz = False, 
										viz_name = "",
										show = False,
										x_feature_index = 0, 
										y_feature_index = 1,
										x_feature_name = "PCA First Feature Component",
										y_feature_name = "PCA Second Feature Component"):
		"""
		Silhouette analysis user called function. This function runs its corresponding
		private method from within the metadata function handler. It also specifies
		variables that need to be de-allocated after the function has finished.

		Args:

			cluster_num:					Number of clusters to test
			viz:							Boolean on whether or not to create visual
			viz_name:						Name tag for the visualization
			show:							Boolean on whether or not to show viz at runtime
			x_feature_index:				Feature index for x-axis on plot
			y_feature_index:				Feature index for y-axis on plot
			x_feature_name:					Name for x-axis on plot
			y_feature_name:					Name for y-axis on plot
		"""

		#Set name metadata
		self.action_name = "silhouette_sub_routine"

		# Initialize the clusterer with n_clusters value and a random generator
		cluster_labels = self.CustomerSeg.SegMethod.cluster_results.iloc[cluster_num - 2,1]

		#Set sub-routine metadata
		self.is_sub_action = True

		#Number of clusters for sub-routine
		self.silhouette_subtest_cluster_num = cluster_num

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		self.silhouette_subtest_avg_score = silhouette_score(self.CustomerSeg.train_data, 
									    						cluster_labels)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(self.CustomerSeg.train_data, 
												  cluster_labels)

		if viz:

			#Create silhouette visualization if requested
			silhouette = silhouette_viz(   self.CustomerSeg, 
										   cluster_labels, 
										   cluster_num, 
										   self.silhouette_subtest_avg_score,
										   sample_silhouette_values,
										   x_feature_index, 
										   y_feature_index,
										   x_feature_name,
										   y_feature_name)

			# Show plot at runtime, if requested
			if show:
				silhouette.show()

			#Save the image and log the image name
			self.CustomerSeg.Log.saveImage(silhouette, "silhouette_viz", self.CustomerSeg.viz_folder_name)
			self.silhouette_viz_filename = self.CustomerSeg.Log.image_name


	#######################################
	# Profiling Functions -- Post analysis
	#######################################

	def __generate_profile(self, 
						   clust_label_preds, 
						   num_clusters, 
						   save = True):
		"""
		Generates a profile of the data, either by looking at the mean
		or median value for different feature components by cluster
		or by looking at cluster centers. This will be the main information 
		that is saved from the clustering analysis to be analyzed in post

		Args:
			clust_label_preds:			Cluster label predictions
			num_clusters:				Number of clusters
			save:						kwarg on whether or not to save data

		"""

		#Set temporary metadata variables
		self.action_name = "generate_profile"
		self.profile_cluster_num = num_clusters
		temp_train_data = None

		#Set the appropriate transformation data, either original or preprocessed train data
		if self.action_dict.get("pca_transform") is not None:
			temp_train_data = copy.deepcopy(self.CustomerSeg.orig_train_data)
		else:
			temp_train_data = copy.deepcopy(self.CustomerSeg.train_data)

		#Set the class labels for the training data
		temp_train_data['class_label'] = clust_label_preds

		#Creates mean and median characteristics for this data
		profile_mean_df = temp_train_data.groupby("class_label").mean()
		profile_median_df = temp_train_data.groupby("class_label").median()

		#Save mean and median data for the cluster
		profile_mean_name = "Persona_Profile_mean_c" + str(num_clusters)
		profile_median_name = "Persona_Profile_median_c" + str(num_clusters)

		#Save the train data cluster allocation in full, along with the cluster centers
		temp_train_data_name = "full_data_cluster_alloc_c" + str(num_clusters)
		cluster_centers = "cluster_centers_c" + str(num_clusters) 

		if save:

			#Save all of the profiles
			self.CustomerSeg.Log.saveProfile(temp_train_data, temp_train_data_name)
			self.CustomerSeg.Log.saveProfile(profile_mean_df, profile_mean_name)
			self.CustomerSeg.Log.saveProfile(profile_median_df, profile_median_name)
			self.CustomerSeg.Log.saveProfile(pd.DataFrame(self.CustomerSeg.SegMethod.elbow_clust_result[0]), cluster_centers)

		#Save the image filename
		self.profile_results_filename = self.CustomerSeg.Log.profile_name

	###########################################################################
	# Private Helper Methods
	###########################################################################

	def __componentVarTheshold(self, cum_var_threshold):
		"""
		Determines the number of PCA components to use based
		on on the cumulative variance threshold and 
		the cumulative variance results from PCA_fit
		
		Args:
			
			cum_var_threshold:

		Returns:

			Optimal number of PCA components to use.

		"""

		#Sum of variance explained
		var_sum = 0

		#For each component, how much variance does it explain
		#and add to our total
		for variance in range(len(self.explained_var_rat_array)):

			#Add the additional variance to our total
			var_sum += self.explained_var_rat_array[variance]

			#Stop the loop when we meet our threshold
			if var_sum >= cum_var_threshold:

				#Add one more to make it the "up-to-but-not-included index"
				return variance + 1