###########################################################################
#
# Segmentation Method Object -- Used for implementation
# 
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

#Data Science and predictive libraries
from sklearn.cluster import k_means

#Package specific imports
# None yet


###########################################################################
# Class and constructor
###########################################################################

class KMeans():
	"""
	This is an encapsulation of the sklearn K-means clustering algorithm.
	This class is purely for implementation. No attributes or class
	references are stored here to maintain a centralized OOP design.

	This code leverages the Sci-Kit learn machine learning package. More
	Information can be found here:

	http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

	Attributes:

				### Information pertaining to class object interation ###


				### Information pertaining to class execution ###

			num_clusters:		Number of clusters 
			max_iter:			Maximum number of centroid re-assignments
								before timeout.
			name:				Name of clustering method (for loggging).
								Set to K_Means here

	"""

	def __init__(self, 
				 max_iter = 30):
		"""
		Constructor for K-means clustering algorithm. Purely implementation.

		Kwargs:	

			max_iter: 		Maximum centroid re-assignments before timeout.
							Set to sklearn default

		"""

		#Set inputs and initialize variables
		self.num_clusters = None
		self.max_iter = max_iter
		self.name = "K_Means"


	###########################################################################
	# Orchestrator
	###########################################################################

	###########################################################################
	# Public Methods
	###########################################################################

	def cluster(self, data, num_clusters, random_state = 42):
		"""
		K_means clustering algorithm implementation process. 		

		Args:

			data:				Train data to cluster
			num_clusters:		Number of clusters chosen to test
			random_state:		Random seed for execution. Set by master object


		Returns:

			K_means clustering object, holding all output and metadata.

		"""

		#Return k_means clustering object
		return k_means(data, 
					   n_clusters = num_clusters, 
					   max_iter = self.max_iter,
					   random_state = random_state)


	###########################################################################
	# Private Helper Methods
	###########################################################################

	