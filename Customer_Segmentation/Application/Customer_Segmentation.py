###########################################################################
#
# K-Means Clustering Implementation Object
# **Initial K-means exploration and analysis done here**
#
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import os
import datetime as dt 
import pandas as pd
from pathlib import Path 

#Package specific imports
from Preprocessing import Preprocess
from Seg_Method import SegMethod
from K_Means import KMeans 
from Seg_Viz import *
from Logger import Log

###########################################################################
# Full test execution orchestrators
###########################################################################

#K-means orchestration fpr SSB
def UI_KMeans_Orch(
				 train_data,
				 orig_data,
				 cluster_range,
				 silhouette_analysis = False,
				 silhouette_cluster_range = range(0,0),
				 train_col_names = None, 
				 x_feature_index = 0,
				 y_feature_index = 1,
				 viz = True,
				 show = False,
				 viz_name = "",
				 test_name = ""):
	"""
	Orchestration package for running a full K-means
	clustering analysis. This function can usually only be
	run after some exploratory analysis has been completed.

	Args:

		train_data:						Training data for clustering
		orig_data:						Original data for clustering, may have more than train
		cluster_range:					Range of clusters tested
										(for the elbow test)
		silhouette_cluster_range:		Range of clusters for silhouette analysis
		train_col_names:				Train data column names
		x_feature_index:				X-axis feature index
		y_feature_index:				Y-axis feature index
		viz:							Boolean on if a visual is wanted
		show:							Do you want to see the viz at runtime
		viz_name:						Visualization nametag for all graphs
		test_name:						Name of the test you are running
	"""

	#Make directory on the users desktop
	segmentation_folder_name = "Customer-Segmentation-Test"  + str(dt.datetime.now().strftime("_%Y-%m-%d_%H.%M.%S"))
	os.makedirs(str(Path.home()) + "\\Desktop\\" + segmentation_folder_name)

	#Make the log
	log = Log("Master-Log", "Preprocess-Log", "SegMethod-Log", directory = str(Path.home()) + "\\Desktop\\" + segmentation_folder_name)
	
	print("\nData\n")
	#Strip and replace off any spaces
	test_name = test_name.strip().replace(" ","_")

	#Initialize customer segmentation test
	test = CustomerSegmentation(Method = KMeans(), 
								data = train_data,
								orig_data = orig_data,
								log = log, 
								test_name = test_name)

	# Set train data and class labels
	test.Preprocess.set_train_data(train_data)

	print("\nPCA\n")
	# Conduct PCA, fit and transformation
	test.Preprocess.PCA_fit(viz = viz, viz_name = viz_name, show = show)
	test.Preprocess.PCA_transform()

	print("\nElbow Chart Analysis\n")
	#Conduct elbow chart analysis
	test.SegMethod.elbow_chart_test(cluster_range, viz = viz,show = show, viz_name = viz_name, profile = True)

	if silhouette_analysis:
		print("\nSilhouette Analysis\n")
		#Conduct Silhouette analysis
		test.Preprocess.silhouette_analysis(silhouette_cluster_range, viz = viz, viz_name = viz_name, show = show)

	print("\nLog Saving\n")
	#Save Preprocess and Method logs
	test.Preprocess.PreprocessLog.savePreprocessLog()
	test.SegMethod.MethodLog.saveMethodLog()

	#Add final masterlog record
	log.addMasterLogRecord(test)
	log.saveMasterLog()


#K-means orchestration fpr SSB
def Client_K_Means_Orch(log,
				 train_data,
				 orig_data,
				 cluster_range,
				 silhouette_analysis = False,
				 silhouette_cluster_range = range(0,0),
				 train_col_names = None, 
				 x_feature_index = 0,
				 y_feature_index = 1,
				 viz = False,
				 show = False,
				 viz_name = "",
				 test_name = ""):
	"""
	Orchestration package for running a full K-means
	clustering analysis. This function can usually only be
	run after some exploratory analysis has been completed.

	Args:

		train_data:						Training data for clustering
		orig_data:						Original data for clustering, may have more than train
		class_label:					Class labels
										(What cluster they are actually 
										a part of?)
		cluster_range:					Range of clusters tested
										(for the elbow test)
		silhouette_cluster_range:		Range of clusters for silhouette analysis
		train_col_names:				Train data column names
		x_feature_index:				X-axis feature index
		y_feature_index:				Y-axis feature index
		viz:							Boolean on if a visual is wanted
		show:							Do you want to see the viz at runtime
		viz_name:						Visualization nametag for all graphs
		test_name:						Name of the test you are running
	"""


	
	print("\nData\n")
	#Strip and replace off any spaces
	test_name = test_name.strip().replace(" ","_")

	#Initialize customer segmentation test
	test = CustomerSegmentation(Method = KMeans(), 
								data = train_data,
								orig_data = orig_data,
								log = log, 
								test_name = test_name)

	# Set train data and class labels
	test.Preprocess.set_train_data(train_data)

	print("\nPCA\n")
	# Conduct PCA, fit and transformation
	test.Preprocess.PCA_fit(viz = viz, viz_name = viz_name, show = show)
	test.Preprocess.PCA_transform()

	print("\nElbow Chart Analysis\n")
	#Conduct elbow chart analysis
	test.SegMethod.elbow_chart_test(cluster_range, viz = viz,show = show, viz_name = viz_name, profile = True)

	if silhouette_analysis:
		print("\nSilhouette Analysis\n")
		#Conduct Silhouette analysis
		test.Preprocess.silhouette_analysis(silhouette_cluster_range, viz = viz, viz_name = viz_name, show = show)

	print("\nLog Saving\n")
	#Save Preprocess and Method logs
	test.Preprocess.PreprocessLog.savePreprocessLog()
	test.SegMethod.MethodLog.saveMethodLog()

	#Add final masterlog record
	log.addMasterLogRecord(test)




#K-means orchestration with
def Demo_K_Means_Orch(log,
				 train_data,
				 class_label,
				 cluster_range,
				 silhouette_cluster_range,
				 train_col_names = None, 
				 x_feature_index = 0,
				 y_feature_index = 1,
				 viz = False,
				 show = False,
				 viz_name = "",
				 test_name = ""):
	"""
	Orchestration package for running a full K-means
	clustering analysis. This function can usually only be
	run after some exploratory analysis has been completed.

	Args:

		train_data:						Training data for clustering
		class_label:					Class labels
										(What cluster they are actually 
										a part of?)
		cluster_range:					Range of clusters tested
										(for the elbow test)
		silhouette_cluster_range:		Range of clusters for silhouette analysis
		train_col_names:				Train data column names
		x_feature_index:				X-axis feature index
		y_feature_index:				Y-axis feature index
		viz:							Boolean on if a visual is wanted
		show:							Do you want to see the viz at runtime
		viz_name:						Visualization nametag for all graphs
		test_name:						Name of the test you are running
	"""

		
	#Strip and replace off any spaces
	test_name = test_name.strip().replace(" ","_")

	#Initialize customer segmentation test
	test = CustomerSegmentation(Method = KMeans(), 
								data = train_data,
								log = log, 
								test_name = test_name)


	# Set train data and class labels
	test.Preprocess.set_train_data(train_data, 
								   col_names = train_col_names)
	test.Preprocess.set_class_label(class_label)

	# Conduct PCA, fit and transformation
	test.Preprocess.PCA_fit(viz = viz, viz_name = viz_name, show = show)
	test.Preprocess.PCA_transform()


	if viz:
		#Create cluster plot visualization if requested
		cluster_plot = cluster_viz(test.train_data, test.class_label, x_feature_index = x_feature_index, y_feature_index = y_feature_index)
		
		#Show the plot at runtime if requested
		if show:
			cluster_plot.show()

		#Save the image
		test.Log.saveImage(cluster_plot, "cluster_plot", test.viz_folder_name)

	#Conduct elbow chart analysis
	test.SegMethod.elbow_chart_test(cluster_range, viz = viz,show = show, viz_name = viz_name, profile = True)

	#Conduct Silhouette analysis
	#test.Preprocess.silhouette_analysis(silhouette_cluster_range, viz = viz, viz_name = viz_name, show = show)

	#Save Preprocess and Method logs
	test.Preprocess.PreprocessLog.savePreprocessLog()
	test.SegMethod.MethodLog.saveMethodLog()

	#Add final masterlog record
	log.addMasterLogRecord(test)


###########################################################################
# Class and constructor
###########################################################################

class CustomerSegmentation():
	"""
	This is a dynamic class built around customer segmentation. It allows for different
	Clustering methods to me implemented in order to achieve segmentation. The class will take
	a dataset and other meta-parameters as input. Then it will orchestrate the execution of 
	different tasks implemented by its sub-packages.

	Attributes:

				### Information pertaining to class object interation ###

		Preprocess:					Preprocessing object
		SegMethod: 					Segmentation method object
		Log: 						Logging object

				### Information pertaining to class execution ###

		master_log_filename:		File name of the master log created.
		random_state:				Seed for random number generator
		train_data:					Train data that will be used for analysis
		orig_train_data:			Original train data given as input
		class_label:				Class label data (if supervised problem)

				### Information pertaining to class metadata and logging ###

		execution_date_start:		Start date of execution. Changed for each new action
		execution_time_start:		Start date of execution. Changed for each new action 
		test_name:					Name of the clustering method used
		viz_folder_name:			File name of the Method log created.
		
	"""

	def __init__(self,
				 Method = KMeans(),
				 data = pd.DataFrame(),
				 orig_data = pd.DataFrame(),
				 log = None,
				 test_name = "",
				 random_state = 42):
		"""
		Constructor for LinearInversion class object. No parameters are given. 
		Necessary information from the estimator class is provided later in the
		execution by QST_sim.

		Args:

			data:							Raw data input
			method:							Implementation method class (e.g. KMeans)
			log:							Master Log object reference
			test_name:						Name of the test being run
			random_state:					Seed for random number generator

		"""

		#Date and time metadata
		self.execution_date_start = dt.datetime.now().date()
		self.execution_time_start = dt.datetime.now().time().strftime("%H.%M.%S")

		#Set test name
		self.test_name = test_name

		#Attribute user inputs to object
		self.data = data 
		
		self.Log = log 
		self.Log.test_number += 1

		#Initialize a preprocessing object
		self.Preprocess = Preprocess(self)

		#Initialize Segmentation Method Object
		self.SegMethod = SegMethod(self, Method = Method)

		#Set random state to default input
		self.random_state = random_state

		#Visualization folder name
		self.viz_folder_name = self.test_name + "_Visualizations"

		#Set original data
		self.orig_train_data = orig_data

		#Initialize all data storage variables
		self.train_data = None
		self.class_label = None


	###########################################################################
	# Function Orchestrator
	###########################################################################
	
	def functionHandler(self, 
						Obj, 
						func, 
						inputs, 
						vars_to_deallocate):
		"""
		This function handler acts as a metadata decorator object. It takes a function
		and its necessary inputs as its own input, as well as a list of variables to
		de-allocate from the Method object when the execution terminates.

		All functions will run through this handler so logging can be done efficiently
		and in a centralized way. TODO: integrate this function into the CustomerSeg
		object so that two of these are not necessary. Preprocessing has its own
		functionHandler.

		Args:

			Obj:					Object to be referenced to find function
			func:					Input function to be executed
			inputs:					"func" necessary inputs as a kwarg dictionary
			vars_to_deallocate:		List of variables to de-allocate

		"""

		#Start date and time metadata
		Obj.execution_date_start = dt.datetime.now().date()
		Obj.execution_time_start = dt.datetime.now().time().strftime("%H.%M.%S")
		utc_start_time = dt.datetime.utcnow()

		#Avoid duplicate references to self
		del inputs['self']

		# Run the function with its necessary inputs
		# The ** de-references the dictionary and assigns
		# everything to the appropriate kwargs
		func(**inputs)

		#Document this action was taken
		Obj.action_dict[Obj.action_name] = 1

		#Close out date and time metadata
		Obj.execution_duration_sec += (dt.datetime.utcnow() - utc_start_time).total_seconds()

		#Add Method Log or Preprocess Log record
		if Obj.__class__.__name__ == "SegMethod":

			#Add to number of preprocess actions
			Obj.num_method_actions += 1
			Obj.MethodLog.addMethodRecord(Obj)

		elif Obj.__class__.__name__ == "Preprocess":

			#Add to number of preprocess actions
			Obj.num_preprocess_actions += 1
			Obj.PreprocessLog.addPreprocessRecord(Obj)

		#De-allocate variables
		for var in vars_to_deallocate:
			del Obj.__dict__[var]		 
					 

	###########################################################################
	# Public Methods -- Shell methods to speed up process
	###########################################################################

	###########################################################################
	# Private Helper Methods -- Also to speed up process
	###########################################################################

	