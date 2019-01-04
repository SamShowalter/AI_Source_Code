###########################################################################
#
# Logger file for collecting execution metadata and results 
# 
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################

###########################################################################
# Module and library imports
###########################################################################

import datetime as dt 
import pandas as pd 
import os
from pathlib import Path

###########################################################################
# Class and Constructor
###########################################################################

class Log():
	"""
	Log object corresponds to a test in three ways. A Master_Log row can be thought of
	as a snapshot of an entire execution, from start to finish. All data collected 
	is stored at a summary level in this log. One Master_Log row corresponds to a 
	single test, and naturally more than one test can be stored in a master log.
	Additionally, Preprocess_Logs store the granular information gathered when preprocessing. 
	Each process log file corresponds to a SINGLE test. Each Master_Log row will include the file
	name of the corresponding Preprocess_Log file. All preprocess_log files are stored in folders relative
	to the master_log which housing the execution summary. One folder corresponds to one Master_Log. 
	The same schema applies to method_logs. Each test utilizes one clustering method, and one method_log 
	file pertains to one row in the Master_Log.

	Attributes:

			### Log Names ###

		master_log_name:					Name provided for the master log
		preprocesss_log_name:				Name provided for the preprocess log
		method_log_name:					Name provided for method execution log

			### Logs ####

		master_log:							Dataframe corresponding to master_log data (manually initialized)
		preprocess_log:						Dataframe corresponding to preprocess_log data (provided)
		method_log:							Dataframe corresponding to method_log data (clustering execution data)
	

			### Other Variables ###

		image_name:							Temporary variable that names the image being saved.
											Overwritten as new images are saved
		test_number:						The number of the test currently being logged
											Used for metadata tracking
		directory:							Directory where all log data will be stored 
											(don't forget to add escape characters when necessary)



	"""

	def __init__(self, master_log_name = "Not Applicable", 
					   preprocess_log_name = "Not Applicable", 
					   method_log_name = "Not Applicable",
					   master_col_names = 
									   		#Execution Metadata
										    [["Execution_Date",
											"Execution_Time",
											"Execution_Duration_Sec",
											"Preprocess_Duration_Sec",
											"SegMethod_Duration_Sec",

											# Test number information
											"Test_Number",

											## Preprocess and Method  Metadata ##
											"Num_Preprocess_Actions",

											## SegMethod metadata info ##
											"Num_SegMethod_Actions",

											# Preprocess steps boolean array 
											"Set_Train_Data",
											"Set_Class_Label",
											"Set_Train_Col_Names",
											"PCA_Fit",
											"PCA_Transform",
											"Silhouette_Analysis",

											## Segmentation Method information ##

											#What method was used
											"Segmentation_Method",

											#K-means clustering info
											"Elbow_Chart_Test",
											"Single_Cluster_Run",

											#Visualization folder name
											"Viz_Folder_Name",

											#Preprocess log file
											"Preprocess_Log_Filename",

											#Method log file
											"SegMethod_Log_Filename",
											
											#Directory where data saved
											"Directory"]],

					   preprocess_col_names = #Preprocess column names, metadata
					   						[["Preprocess_Execution_Date",
											  "Preprocess_Execution_Time",
											  "Preprocess_Duration_Sec",

											  #Action name
											  "Action_Name",

											  #Is it a sub-action?
											  "Sub_Action",

											  #Test ratio for train test split
											  "Test_Ratio",

											  #Information for PCA
											  "Cum_Var_Threshold",
											  "PCA_Cum_Var_Viz_Filename",
											  "Num_PCA_Components",
											  "Explained_Var_Rat_Array",

											  #Profiling information
											  "Profile_Cluster_Num",
											  "Profile_Results_Filename",

											  #Silhouette Analysis header
											  "Silhouette_Cluster_Range_Start",
											  "Silhouette_Cluster_Range_End",
											  "Silhouette_X_Feature_Index",
											  "Silhouette_Y_Feature_Index",

											  #Information for silhouette subtests
											  "Silhouette_Subtest_Cluster_Num",
											  "Silhouette_Viz_Filename",
											  "Silhouette_Subtest_Avg_Score"

											]],

						method_col_names = #Segmentation method column names
											[["SegMethod_Execution_Date",
											  "SegMethod_Execution_Time",
											  "SegMethod_Duration_Sec",

											  #ACtion name for method
											  "Action_Name",

											  #Is it a sub-routine
											  "Is_Sub_Action",

											  #Specific information about single cluster run
											  "Single_Cluster_Num",
											  "Single_Sqr_Dist",

											  #Elbow chart analysis header
											  "Elbow_Cluster_Range_Start",
											  "Elbow_Cluster_Range_End",
											  "Elbow_Viz_Filename",

											  #Elbow chart sub-tests
											  "Elbow_Subtest_Cluster_Num",
											  "Elbow_Subtest_Sqr_Dist"
											  ]],

					   directory = (str(Path.home()) + "\\Desktop")):

		"""
		Constructor for Log object. Creates dataframe logs and sets the correct directory

		Args:

				### Log Names ###

			master_log_name:					Name provided for the master log
			preprocess_log_name:				Name provided for the preprocess log
			method_log_name:					Name provided for the method log

				### Log Column Names ###

			master_col_names:					Column names for the master log
			preprocess_col_names:				Column names for the preprocess log
			method_col_names:					Column names for the method log

				### Other Variables ###

			test_number:						Number of the test being logged
			directory:							Directory where all log data will be stored 
												(don't forget to add escape characters when necessary)
		"""

		#Set the directory to store all data output
		self.directory = directory

		#Initialize test number
		self.test_number = 0

		#Log names
		self.master_log_name = master_log_name + str(dt.datetime.now().strftime("_%Y-%m-%d_%H.%M.%S"))
		self.preprocess_log_name = preprocess_log_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3])
		self.method_log_name = method_log_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3])

		#Two DataFrame logs for performance and data collection
		self.master_log = pd.DataFrame(columns = master_col_names)
		self.preprocess_log = pd.DataFrame(columns = preprocess_col_names)
		self.method_log = pd.DataFrame(columns = method_col_names)


	###########################################################################
	# Public methods that facilitate execution of logger
	###########################################################################

	def addMasterLogRecord(self, Test):
		"""
		Adds a record to the master log. (One record = one test)

		Args:

			test:					test (CustomerSegmentation) object containing all data from test
			
		"""

		#Create a new record
		new_record_df = pd.DataFrame(
								      [[
								       #Test Execution Information
								       Test.execution_date_start,
								       Test.execution_time_start,
								       Test.Preprocess.execution_duration_sec + Test.SegMethod.execution_duration_sec, #Total execution time
								       Test.Preprocess.execution_duration_sec,
								       Test.SegMethod.execution_duration_sec,

								       #Test Number
								       self.test_number,

								       #Preprocessing information
								       Test.Preprocess.num_preprocess_actions,

								       #Number of SegMethod actions
								       Test.SegMethod.num_method_actions,

								       #Boolean Checks for preprocessing steps -- may not exist
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['set_train_data']"),
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['set_class_label']"),
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['set_train_col_names']"),
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['pca_fit']"),
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['pca_transform']"),
								       self.__try_exists(Test, "Obj.Preprocess.action_dict['silhouette_analysis']"),

								       #Segmentation Method Information
								       Test.SegMethod.seg_method_name,										#Method name

								       #K-means clustering information -- may not exist
								       self.__try_exists(Test, "Obj.SegMethod.action_dict['elbow_chart_test']"), #Elbow chart test
								       self.__try_exists(Test, "Obj.SegMethod.action_dict['cluster']"),			#Cluster algorithm

								       #Visualization folder name
								       Test.viz_folder_name,

								       #Relevant Preprocess log file
								       Test.Preprocess.PreprocessLogFilename,

								       #Relevant Method log file
								       Test.SegMethod.MethodLogFilename,

								       #Relevant directory
								       Test.Log.directory]],			

									   #Add the Master Log Column Names
									   columns = self.master_log.columns)

		#Add record to the existing dataframe
		self.master_log = pd.concat([self.master_log, new_record_df], axis = 0)
		self.master_log.reset_index(drop = True, inplace = True)

	def addPreprocessRecord(self, Preprocess):
		"""
		Adds a record to the preprocess log. (One record = one preprocess action)

		Args:

			Preprocess:					Preprocess object, where all of the pre-processing occurs.
			
		"""

		#Create new preprocess record
		new_metadata_df = pd.DataFrame(
								      [[
								      	#Preprocess Execution Information
								      	Preprocess.execution_date_start,					#Current date
								       	Preprocess.execution_time_start,					#Current time
								       	Preprocess.execution_duration_sec,					#Preprocess duration in seconds

								       	#Preprocess action name
								       	Preprocess.action_name,

								       	#Is something a sub-routine of a larger function?
								       	self.__try_exists(Preprocess, "Obj.is_sub_action", errorVal = "False"),

								       	#Specific information on train_test_split
								       	self.__try_exists(Preprocess, "Obj.test_ratio"),

								       	#Specific information for PCA_fit and transform
								       	self.__try_exists(Preprocess, "Obj.cum_var_threshold"),			# Variance Threshold
								       	self.__try_exists(Preprocess, "Obj.PCA_cum_var_viz_filename"),	# PCA_fit_viz filename
								       	self.__try_exists(Preprocess, "Obj.num_pca_components"),		# Number of PCA_components chosen
								       	
								       	#This must be a repr() so it can be read back into system if necessary
								       	repr(self.__try_exists(Preprocess, "Obj.explained_var_rat_array")),	#Explained variance ratio array

								       	#Information about Post_Processing Profile filename
										self.__try_exists(Preprocess, "Obj.profile_cluster_num"),
								       	self.__try_exists(Preprocess, "Obj.profile_results_filename"),

								       	#Specific information for Silhouette Analysis header
								       	self.__try_exists(Preprocess, "Obj.silhouette_cluster_range_start"),
								       	self.__try_exists(Preprocess, "Obj.silhouette_cluster_range_end"),
								       	self.__try_exists(Preprocess, "Obj.silhouette_x_feature_index"),
								       	self.__try_exists(Preprocess, "Obj.silhouette_y_feature_index"),

								       	#Specific information for Silhouette Analysis subtests
								       	self.__try_exists(Preprocess, "Obj.silhouette_subtest_cluster_num"),
								       	self.__try_exists(Preprocess, "Obj.silhouette_viz_filename"),
								       	self.__try_exists(Preprocess, "Obj.silhouette_subtest_avg_score")

										]],		

									   #Add the preprocess Log Column Names
									   columns = self.preprocess_log.columns)

		#Add preprocess record to the preprocess_log
		self.preprocess_log = pd.concat([self.preprocess_log ,new_metadata_df], axis = 0)
		self.preprocess_log.reset_index(drop = True, inplace = True)

	#Add record for MLE execution
	def addMethodRecord(self, SegMethod):
		"""
		Adds a record to the method log pertaining to clustering execution.
		(one record = one segmentation method action)

		Args:

			SegMethod:					SegMethod clustering segmentation object
			
		"""

		#Create new SegMethod record
		methodRecord = pd.DataFrame([[#Clustering method execution information
								      	SegMethod.execution_date_start,					#Current date
								     	SegMethod.execution_time_start,					#Current time
								      	SegMethod.execution_duration_sec,				#Model duration in seconds	

								        #Action name for method
      									SegMethod.action_name,

      									#Is something a sub-step of a larger function
								       	self.__try_exists(SegMethod, "Obj.is_sub_action", errorVal = "False"),

								      	#Specific information about single cluster run
								      	self.__try_exists(SegMethod, "Obj.single_cluster_num"),
								      	self.__try_exists(SegMethod, "Obj.single_sqr_dist"),

										#Specific information for Elbow Plot header
										self.__try_exists(SegMethod, "Obj.elbow_cluster_range_start"),
										self.__try_exists(SegMethod, "Obj.elbow_cluster_range_end"),
										self.__try_exists(SegMethod, "Obj.elbow_plot_viz_filename"),

										#Specific information for Elbow Plot subtests
										self.__try_exists(SegMethod, "Obj.elbow_subtest_cluster_num"),
										self.__try_exists(SegMethod, "Obj.elbow_subtest_sqr_dist")

								      ]],

					 columns = self.method_log.columns)

		#Add verbose record to the verbose_log
		self.method_log = pd.concat([self.method_log ,methodRecord], axis = 0)
		self.method_log.reset_index(drop = True, inplace = True)



	def saveMasterLog(self):
		"""
		Saves the master log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""
		#Change working directory for Master Logs
		os.chdir(self.directory)

		#Save the master_log
		self.master_log.to_csv(self.master_log_name + ".csv", sep = ",")

	def savePreprocessLog(self):
		"""
		Saves the preprocess log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""

		#Change working directory for Result Logs
		os.chdir(self.directory)

		#Check to see if a master_log folder has been created
		if not os.path.exists(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number)):
			os.makedirs(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number))

		#Navigate to the appropriate directory
		os.chdir(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number))

		#Save the boot log
		self.preprocess_log.to_csv(self.preprocess_log_name + ".csv", sep = ",")
	
	def saveMethodLog(self):
		"""
		Saves the method log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""


		#Change working directory for Result Logs
		os.chdir(self.directory)

		#Check to see if a master_log folder has been created
		if not os.path.exists(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number)):
			os.makedirs(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number))
			
		#Change the directory to within the method_log
		os.chdir(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number))

		#Save the method log
		self.method_log.to_csv(self.method_log_name + ".csv", sep = ",")


	def saveProfile(self, 
					profile, 
					profile_name):
		"""
		Saves a persona profile: Determines the mean characteristics of each cluster for all
		feature attributes. 

		args:

			profile:		Dataframe representing the profile characteristics for the features
			profile_name:	Name of the profile

		"""

		#Check to see if the profile directory is set. If not, make it
		if not os.path.exists(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\Profiles"):
		 	os.makedirs(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\Profiles")

		#Navigate to the appropriate directory
		os.chdir(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\Profiles")

		#Set the profile name with input name and datetime data
		self.profile_name = profile_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3]) + ".csv"

		#Save the profile
		profile.to_csv(self.profile_name)

	def savePCAComp(self, 
					pca_comp, 
					pca_comp_name):
		"""
		Saves the pca composition for a specific cluster. Intended to show
		the feature importance or impact of features on the definition
		of a cluster 

		args:

			pca_comp:		Dataframe pca composition of the data
			pca_comp_name:	Name for the pca composition name

		"""

		#Check to see if the profile directory is set. If not, make it
		if not os.path.exists(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\PCA_Composition"):
		 	os.makedirs(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\PCA_Composition")

		#Navigate to the appropriate directory
		os.chdir(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\PCA_Composition")

		#Set the profile name with input name and datetime data
		self.pca_comp_name = pca_comp_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3]) + ".csv"

		#Save the profile
		pca_comp.to_csv(self.pca_comp_name)

	def saveImage(self, img, image_name, viz_folder_name):
		"""
		Saves an image passed to the logger from another object.

		Args:

			img:				Matplotlib.pyplot object with figure created
			image_name:			Name image should be saved as
			viz_folder_name:	Folder image should be saved in

		"""
		#Check to see if a master_log folder has been created
		if not os.path.exists(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\" + viz_folder_name):
			os.makedirs(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\" + viz_folder_name)

		#Change directory to correct visualization folder
		os.chdir(self.directory + "\\" + self.master_log_name + "\\Test_" + str(self.test_number) + "\\" + viz_folder_name)

		#Image name with added metadata to make unique
		self.image_name = image_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3]) + ".png"

		#Save the image
		img.savefig(self.image_name)

		#Close the image and de-allocate memory
		img.close('all')

	###########################################################################
	# Private methods 
	###########################################################################

	def __try_exists(self,Obj,inputString, errorVal = "-"):
		"""
		Private method that facilitate the seamless execution of the logger 
		in the face of a very dynamic logging system. Ensures that all variable 
		calls are handled as exceptions and will not cause the execution to error out.

		Args:

			Obj:			Relevant object that needs to be referenced
							We want to see if this object has a certain input var
			inputString:	String representation of the variable we want to access
			errorVal: 		Default value to return on exception

		"""

		try:
			#If something is defaulted to None, set it as an error
			if eval(inputString) is None:
				return errorVal

			#Try to evaluate and return class attribute
			return eval(inputString)

		except Exception as e:

			#If attribute does not exist, return errorVal
			return errorVal


