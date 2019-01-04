###########################################################################
#
# Main method for package testing
# **Initial K-means exploration and analysis done here**
#
# Author: Sam Showalter
# Date: September 6, 2018
#
###########################################################################


###########################################################################
# Library Imports
###########################################################################

#Data science Libraries
from sklearn import datasets
import numpy as np
import pandas as pd
import datetime as dt

#Import custom Customer Segmentation package
from Customer_Segmentation import *
from Logger import Log 
from K_Means import KMeans 
from Seg_Viz import *

###########################################################################
# Preprocessing -- none for this test -- just grabbing dataset
###########################################################################

iris = datasets.load_iris()
train = pd.DataFrame(iris.data) 
test = pd.DataFrame(iris.target)

###########################################################################
# Test the execution
###########################################################################

if __name__ == "__main__":
	
	#Create Master Log
	MasterLog = Log("Master-Log", "Preprocess-Log", "SegMethod-Log")

	#Iterate and run several different tests
	for i in range(1):
		Client_K_Means_Orch(
					 # Reference to the master log
					 MasterLog,

					 #Training data
					 train,

					 #Original data (can have more than train data)
					 train,

					 #Elbow chart cluster range
					 range(2,12),

					 #Silhouette analysis cluster range
					 silhouette_analysis = True,
					 silhouette_cluster_range = range(3,6),

					 #Visualization information (do we want pictures? See them at runtime?)
					 viz = True,
					 show = False,
					 viz_name = "Iris_Data",

					 #Test Name
					 test_name = "Iris_Data")

	#Save the master log after all tests have been run
	MasterLog.saveMasterLog()


	########################################################################################
	# Miscellaneous tests

	# test = CustomerSegmentation(iris, KMeans())

	# test.Preprocess.set_train_data(iris.data)
	# test.Preprocess.set_train_col_names(["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])
	# test.Preprocess.set_class_label(iris.target)


	# #print(test.train_data.head())

	# test.Preprocess.PCA_fit(viz = False)
	# test.Preprocess.PCA_transform()

	# # #print(test.train_data.head())
	# cluster_viz(test.train_data, test.class_label, x_feature_index = 0, y_feature_index = 1)

	# test.SegMethod.elbow_chart_test(range(2,12), viz = False, viz_name = "Iris Data")

	# test.Preprocess.silhouette_analysis(range(2,3))



	########################################################################################
	# Evaluation code for log enhancements

	# a = np.array([1,2,3,4,5])

	# b = str(np.array_repr(a))

	# c = eval("np." + b)

	# print(c)

	# print(c.sum())

	#print(np.array_repr(a))

	########################################################################################










