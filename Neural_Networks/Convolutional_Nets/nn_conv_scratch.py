###########################################################################
#
# CONVOLUTIONAL Neural Network Implementation from Scratch
# -- Built for a WMP KTS --
#
# Author: Sam Showalter
# Date: October 24, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import os
import datetime as dt 
import copy
import sys
import pickle as pkl

#Visualization libraries
import matplotlib.pyplot as plt

#Data Science and predictive libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d, fftconvolve
import skimage.measure as measure
#Dataset related imports
import tensorflow as tf
#from sklearn import datasets
mnist = tf.keras.datasets.mnist

###########################################################################
# Data formatting and restructuring for analysis 
###########################################################################


(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
x_train = x_train[:3,:,:]
y_train = y_train[:3]
print(x_train.shape)
# iris = datasets.load_iris()
# train = pd.DataFrame(iris.data) 
# test = pd.Series(iris.target)


# test =(pd.get_dummies(pd.Series(test)))


# x_train, x_test,y_train, y_test = train_test_split(train, test, test_size = 0.33, random_state = 42)

# print(pd.get_dummies(y_train))

# y_train = np.array((pd.get_dummies(pd.Series(y_train))).values)


# print(x_train)
# print(y_train)


###########################################################################
# Class and constructor
###########################################################################
class Convolution2D():

	def __init__(self, 
				 output_channels = 32,
				 kernel_size = (5,5),
				 stride = (1,1),
				 activation = 'relu',
				 hidden = True):
		self.type = 'convolution'
		self.output_channels = output_channels 
		self.kernel_size = kernel_size
		self.stride = stride
		self.hidden = hidden
		self.activation = activation

class MaxPooling():

	def __init__(self,
				 pool_size = (2,2),
				 stride = (2,2)):

		self.type = 'max_pooling'
		self.pool_size = pool_size
		self.stride = stride

class Flatten():

	def __init__(self):
		self.type = 'flatten'

class Dense():

	def __init__(self,
				 num_nodes,
				 activation = 'tanh',
				 output_layer = False):

		self.type = 'dense'
		self.init = False
		self.num_nodes = num_nodes
		self.activation = activation 
		self.output_layer = output_layer


class NN_Conv_Scratch():
	"""
	This class implements convolutional neural networks from scratch.
	Primarily, this class is to be used for teaching purposes, though
	with a small amount of tweaking it could be applied to a client project. Networks
	are optimized with backpropagation.

	Attributes:

		network_name:				Name of the network (usually tied with type of data)
		nerwork_filename:			Filename of compressed network that will be imported
		directory: 					Directory that networks will be saved off to
		
			### Data attributes to train neural network (inputs) ###
		
		train_data:					Training dataset, feature set as a numpy Array
		labels:						Output labels, usually as one-hot encoded array
									Regression would NOT use a one-hot encoded array
		train_verbose:				If someone wants to see the progress and loss/accuracy during training
		update_learn_rate:			Boolean on if the learning rate should be updated while training occurs

			### Neural Network structure attributes ###

		num_inputs:					Number of input features into the network
		num_outputs:				Number of class labels that can be predicted
		num_hidden_layers:			Number of hidden layers in the network (must be at least 1)
		hidden_layer_nodelist:		The number of nodes to have in each hidden layer
									Number of nodes is sequential with the layers


			### Activation and backpropagation attributes ###

		activation_dict:			Dictionary storing all activation functions and derivatives
		activate:					Chosen activation function
		derivActivate:				Chosen actication function derivative
		epsilon:					Learning rate for gradient descent
		reg_lambda:					Regularization strength

			### Network output and evaluation attributes ###

		log_loss:					Log loss function to determine entropy loss (how good is fit?)
		log_loss_dif:				Difference from last log_loss to new one (is it decreasing?)
		results:					Results from feed forward process in neural network
									This is a one-hot encoded array usually. Unless specified otherwise
		preds:						Predictions derived from one-hot encoded results array
		accuracy:					Accuracy calculation

	
	"""

	def __init__(	self, 
				 	train_data,
				 	labels,
				 	network,
					num_outputs,
					epsilon = 0.001,
					reg_lambda = 0.001,
					network_name = "network",
					network_filename = None,
					update_learn_rate = False,
					directory = "C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\BA_Source_Code\\Neural_Networks\\output"):
		"""
		Constructor for Neural Network from scratch class. Given the inputs, the neural network is
		created and the appropriate activation functions are chosen.

		Args:
				
				network_name:				Name of the network (usually tied with type of data)
				nerwork_filename:			Filename of compressed network that will be imported
				directory: 					Directory that networks will be saved off to

					### Data attributes to train neural network (inputs) ###
				
				train_data:					Training dataset, feature set as a numpy Array
				labels:						Output labels, usually as one-hot encoded array
											Regression would NOT use a one-hot encoded array

					### Neural Network structure attributes ###

				num_outputs:				Number of class labels that can be predicted

					### Activation and backpropagation attributes ###

				activation_function:		Activation function keyword. Corresponds to key in activation_dict
				epsilon:					Learning rate for gradient descent
				reg_lambda:					Regularization strength

		"""

		#Initialize neural network
		self.directory = directory
		self.network_name = network_name
		self.network = network

		#Activation dictionary; stores all activation functions
		self.activation_dict = {"tanh": self.tanh,
								"derivtanh": self.derivTanh,
								"relu": self.relu,
								"derivrelu": self.derivRelu}

		#Set train data and output labels
		self.full_train_data = train_data
		self.train_data = train_data
		self.full_labels = labels

		#Initialize log_loss and log loss difference
		self.log_loss = np.Inf
		self.log_loss_dif = 0

		# Gradient descent parameters 
		self.epsilon = epsilon 			# learning rate for gradient descent
		self.reg_lambda = reg_lambda 	# regularization strength

		#Network dimension information
		self.num_outputs = num_outputs

		#Determine if you should update learn rate
		self.update_learn_rate = False

		#Create the neural network. Initialize random weights or import
		if network_filename is None:
			self.make_network()

		#Load the network if one already exists
		else:
			print("Loading Network")
			self.load_network(network_filename)

###########################################################################
# Orchestration functions for training and testing
###########################################################################

	def train(		self,
					num_epochs = 10000,
					epoch_update_cadence = 1000,
					verbose = False,
					update_learn_rate = False,
					gradient_method = "batch",						#Methods include batch, minibatch, sgd
					batch_size = None):				
		"""
		Orchestration method for training the neural network. 
		"""
		self.update_learn_rate = update_learn_rate
		self.minibatch_size = batch_size
		self.gradient_method = gradient_method
		num_shuffles = 1

		if (self.gradient_method == "batch"):
			self.train_data = self.full_train_data
			self.labels = self.full_labels

		elif (self.gradient_method == "minibatch"):
			num_shuffles = int(np.ceil(self.full_train_data.shape[0] / self.minibatch_size))
			
		for i in range(num_epochs):

			if (self.gradient_method == 'minibatch'):
				rand_order = np.random.permutation(self.full_train_data.shape[0])
				batch_train_shuffle = self.full_train_data[rand_order]
				batch_label_shuffle = self.full_labels[rand_order]

				for j in range(num_shuffles - 1):

					self.train_data = batch_train_shuffle[j*self.minibatch_size: min((j+1)*self.minibatch_size, 
																								self.full_train_data.shape[0])]
					self.labels = batch_label_shuffle[j*self.minibatch_size: min((j+1)*self.minibatch_size, 
																								self.full_train_data.shape[0])]
					self.feed_forward()
					self.backpropagate()

			else:
				self.feed_forward()
				self.backpropagate()

			#If verbose, update user at specified epoch cadence
			if i % epoch_update_cadence == 0:
				print("Iteration: %s\nLoss: %s\nAccuracy: %s\n"%(i,self.log_loss, self.accuracy))
				sys.stdout.flush()

	
	def predict(	self, 
					test_input, 
					test_output):
		"""
		Predicts the output for a set of inputs, given the weights determined by
		the neural network. This really should only be run after training finishes.

		Args:

			test_input:				Input data (features) to be used for input prediction
			test_output:			Actual class labels to be compared to output
		"""
		self.feed_forward(test = True, test_input = test_input, test_output = test_output)


###########################################################################
# Network creation and forward propagation
###########################################################################

	def make_network(self):
		"""
		Create the neural network structure. All weights and biases are initialized
		as random weights and zeros, respectively. Each layer is conditionally created 
		differently
		"""
		self.num_layers = len(self.network)
		for layer_index in range(len(self.network)):

			layer = self.network[layer_index]

			if layer.type == "convolution":
				if not layer.hidden:
					layer.weights = np.random.rand(layer.output_channels, 1, layer.kernel_size[0], layer.kernel_size[1])
					
				else:
					layer.weights = np.random.rand(layer.output_channels, self.network[layer_index - 2].output_channels, layer.kernel_size[0], layer.kernel_size[1])
					
				layer.biases = np.zeros((1,layer.output_channels))
			elif layer.type == "max_pooling":
				continue
			elif layer.type == "flatten":
				continue
			elif layer.type == "dense":
				continue

	def feed_forward(	self, 
						test = False, 
						test_input = None, 
						test_output = None):
		"""
		Feed forward process of pushing feature input through the network to get
		output probabilities and predictions. This is also called as the "predict"
		function for final output, and is the reason we feed in test input and test output.

		Args:
				test:			Boolean to determine if this is a prediction after training
				test_input:		test_input to replace training data if test == True
				test_output:	test_output to replace testing labels if test == True

		"""

		#Initialize throughput for this analysis
		throughput = None

		#Determine if this is for training or prediction, and update
		if test:
			data = test_input
			test_output = test_output
		else:
			data = self.train_data

		#For all layers, including output layer
		for i in range(self.num_layers):

			layer = self.network[i]

			#If this is the input layer, which will always be a convolutional layer here
			if (i == 0):
				throughput = self.relu(np.array([np.array(
										[fftconvolve(self.train_data[j,:,:], layer.weights[k,0,:,:], mode = 'valid') + 
										layer.biases[0,k] 
										for k in range(layer.weights.shape[0])]) 
										for j in range(self.train_data.shape[0])]))

				# throughput = np.array([np.array(
				# 						[convolve2d(self.train_data[j,:,:], layer.weights[k,:,:], mode = 'valid') + 
				# 						layer.biases[0,k] 
				# 						for k in range(layer.weights.shape[0])]) 
				# 						for j in range(self.train_data.shape[0])])
				
				# throughput = self.relu(throughput)
				print(throughput.shape)
				#print(throughput.sum())

			#If this is a hidden layer
			else:
				if layer.type == "convolution":
					throughput = np.squeeze(self.relu(np.array([np.array(
										[fftconvolve(throughput[j,:,:,:], layer.weights[k,:,:,:], mode = 'valid') + 
										layer.biases[0,k] 
										for k in range(layer.weights.shape[0])]) 
										for j in range(throughput.shape[0])])))
					print(throughput.shape)
					#print(throughput.sum())

				elif layer.type == "max_pooling":
					throughput = np.array( [np.array([measure.block_reduce(throughput[k,j,:,:], layer.pool_size, np.max) 
											for j in range(throughput.shape[1])])
											for k in range(throughput.shape[0])])

					#First dim = Number of obs.
					#Second dim = number of output channels.
					#Third dim = Image - y
					#Fourth dim = Image - x

					# So is backprop conducted in the same way as MLPs, and how is it transferred appropriately 
					# to kernels? Math aside, is the error propagation backward a similar concept?
					# backprop is most important in the actual conv layer - so each channel is trying
					# to find a relevant pattern that assists in classification
					# example, one convo slice (i think of them as filters) could be a vertical line

					# 0001000
					# 0001000
					# 0001000
					# 0001000
					# # would be a relu parameter converged on finding a vertical line
					# # this would be blob of red
					# 00000
					# 01110
					# 00000
					# in the R channel of RGB - does that help?

					# Take a look at this...

					print(throughput.shape)
					#print(throughput.sum())

				elif layer.type == "flatten":
					throughput = throughput.reshape(self.train_data.shape[0], -1)
					print(throughput.shape)
					#print(throughput.sum())

				elif layer.type == "dense":
					if not layer.init:
						layer.weights = np.random.rand(throughput.shape[1], layer.num_nodes)
						layer.biases = np.random.rand(1,layer.num_nodes)
						layer.init = True

					throughput = np.matmul(throughput, layer.weights) + layer.biases

					if layer.output_layer:
						throughput = self.softmax(throughput)

					print(throughput.shape)
					#print(throughput.sum())

		# Determine the output probabilities of each class with softmax
		self.probs = throughput

		sys.exit(1)

		#Determine the predictions by converting the probabilities into output
		self.preds = self.probs.argmax(axis = 1)

		#Find the accuracy of this run
		self.get_accuracy(test = test, test_output = test_output)

		#Determine the cross entropy loss
		self.cross_entropy_loss(test = test, test_output = test_output)

		#Update the learning rate (epsilon)
		if self.update_learn_rate:
			self.update_learning_rate()



###########################################################################
# Backpropagation functions
###########################################################################

	def backpropagate(self):
		"""
		Implements backpropagation on the Artifical Neural Network. This process is dynamic for the
		number of hidden layers, the number of nodes in each layer, and the chosen activation function.
		A scaling term defined by the number of examples (rows) is applied to all weight adjustments.
		Memoization is used to make sure the "delta" changes efficiently and correcly apply to each layer.
		"""

		#Create memoization delta. Starts as difference in expected and actual results
		#Develop scaling term
		self.network["delta"] = self.probs - self.labels
		scaling = 1 / len(self.probs)
		
		#Moving backward through network, excluding first layer (0th layer)
		for i in range(self.num_hidden_layers, 0, -1):

			#Determine changes to weights and biases for specific layer 
			self.network["dw" + str(i)] = self.network["a" + str(i - 1)].T.dot(self.network["delta"]) * scaling
			self.network["db" + str(i)] = np.sum(self.network["delta"], axis = 0, keepdims =True) * scaling

			#Update the delta function with the new values for next layer
			self.network["delta"] = np.multiply(np.matmul(self.network["delta"], self.network["w" + str(i)].T), 
												self.derivActivate(self.network["a" + str(i - 1)]))
		
		#Final layer adjustment, tied to input data
		self.network["dw0"] = np.dot(self.train_data.T,self.network["delta"]) * scaling
		self.network["db0"] = np.sum(self.network["delta"], axis=0, keepdims = True) * scaling

		#Update the weights for all of the layers
		self.update_weights()

	def update_weights(self):
		"""
		Update all the weights of the network after backpropagation has been conducted.
		This process adjusts all weights by the epsilon level.
		"""

		#For all layers, including the output layer (hence the +1)
		for i in range(self.num_hidden_layers + 1):

			# Finalize the delta term for each layer and add regularization strength
			self.network["dw" + str(i)] += self.reg_lambda * self.network["w" + str(i)]

			# Gradient descent parameter update
			self.network["w" + str(i)] += -self.epsilon * self.network["dw" + str(i)]
			self.network["b" + str(i)] += -self.epsilon * self.network["db" + str(i)]


	def update_learning_rate(self):
			"""
			Method used to update the epsilon learning rate while the network is
			training. This can allow for a better overall learning process as the gradient level decreases
			as training progresses, requiring lower or higher gradient descent learning rates.

			This statements are simply estimations on what will improved specific cases, and not an instance
			of statistical reasoning. More nuanced updates will follow in further commits.
			"""

			#If the accuracy is above 90% begin reducing the learning rate
			if self.accuracy > 0.90 and self.log_loss_dif < 0 and self.epsilon > 0.005:
				self.epsilon = self.epsilon / 1.5

			#Continue to decrease this rate as the accuracy exceeds 95%
			if self.accuracy > 0.95 and self.log_loss_dif < 0 and self.epsilon > 0.001:
				self.epsilon = self.epsilon / 1.5

###########################################################################
# Code for determining the optimal learning rate
###########################################################################

	def learning_rate_test(self, starting_rate = 1e-40, num_epochs = 500, step = 1.9):
		self.epsilon = starting_rate

		epsilons = []
		log_loss = []		

		for epoch in range(num_epochs):
			self.feed_forward()
			self.backpropagate()

			epsilons.append(self.epsilon)
			log_loss.append(self.log_loss)

			self.epsilon = min(1.0, self.epsilon*step)

			if epoch % 10 == 0:
				print("Iteration: %s"%(str(epoch)))

			if self.epsilon == 1.0:
				print("Learning Rate Test Finished")
				return epsilons, log_loss

		return epsilons, log_loss


###########################################################################
# Supplemental network functions
###########################################################################

	def softmax(self, 
				throughput):
		"""
		Softmax function to determine pseudo probabilities for output.

		Args:

			throughput: 		Output of the feed-forward network. One-hot encoded. [num_examples x num_features]

		Returns:

			softmax probabilities as a one-hot encoded array. [num_examples x num_features]

		"""
		e_x = np.exp(throughput / np.max(throughput))
		return e_x / e_x.sum(axis = 1)[:,None]

	def cross_entropy_loss(		self, 
								test = False, 
								test_output = None):
	    """
		Determine cross entropy (log) loss for the output. This is used to determine how fast the
		network is learning.

		Reference:
			https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays 
	    
	    """

	    if test:
	    	labels = test_output
	    else:
	    	labels = self.labels

	    #Set the last log_loss so we can track the changes in loss
	    last_log_loss = self.log_loss

	    #Determine log_likelihood, convert it to loss, then update the log_loss_dif
	    log_likelihood = -np.log(self.probs[np.arange(len(self.probs)),labels.argmax(axis = 1)])
	    self.log_loss = np.sum(log_likelihood) / self.probs.shape[0]
	    self.log_loss_dif = last_log_loss - self.log_loss_dif

	def get_accuracy(	self, 
						test = False, 
						test_output = None):
		"""
		Determine the accuracy of the output predictions. This can be provided
		both during training and final predictions.

		Args:

			test:			Boolean determining if this is a post-training prediction
			test_output:	Used in lieu of training output if test == True
		"""

		#If final output predictions are requested after training
		if test:
			self.accuracy = np.sum(self.preds == test_output.argmax(axis = 1)) / len(test_output)

		#If training accuracy is requested
		else:
			self.accuracy = np.sum(self.preds == self.labels.argmax(axis = 1)) / self.probs.shape[0]

	def save_network(self):
		"""
		Pickles and saves the neural network to be used for later if necessary.
		This way you can train a network and save it for a rainy day :).
		"""

		#Change to the correct directory
		os.chdir(self.directory)

		#Create the file name
		filename = (self.network_name + "_i" + str(self.num_inputs) +  "_h" + ".".join([str(i) for i in self.hidden_layer_nodelist]) 
					+ "_o" + str(self.num_outputs) + "_" + str(dt.datetime.now().strftime("_%Y-%m-%d_%H.%M.%S")))

		#Save off the neural network
		with open(filename + '.pickle', 'wb') as network_name:
			pkl.dump(self.network, network_name, protocol=pkl.HIGHEST_PROTOCOL)



	def load_network(	self, 
						filename):
		"""
		Loads in a network so you do not have to initialize and train
		one using random weights.

		Args:

			filename: 			Name of the file to be read in as the network
		"""

		#Change to the correct directory
		os.chdir(self.directory)

		#Read in the network
		with open(filename + ".pickle", 'rb') as network:
			self.network = pkl.load(network)

###########################################################################
# Activation functions, all stored in activation_dict
###########################################################################

	def relu(self,
			 input_arr,):
		return np.maximum(input_arr,0)

	def derivRelu(self, input_arr):
		return np.greater(input_arr, 0).astype(float)

	def tanh(	self, 
				input_arr):
		"""
		Tanh() activation function.

		Args:

			input: Input to be altered by activation function
		"""
		return np.tanh(input_arr)

	def derivTanh(	self, 
					input_arr):
		"""
		Tanh() derivative activation function for backpropagation

		Args:

			input: Input to be altered bywewr activation function for backprop
		"""
		return 1 - np.power(np.tanh(input_arr), 2)


###########################################################################
# Main method for testing
###########################################################################


if __name__ == '__main__':

	nn = NN_Conv_Scratch(x_train, 
						 y_train,
						 [Convolution2D(output_channels = 32, kernel_size = (3,3), stride = (1,1), hidden = False),
						  MaxPooling(pool_size = (2,2), stride = (2,2)),
						  Convolution2D(output_channels = 64, kernel_size = (3,3), stride = (1,1)),
						  MaxPooling(pool_size = (2,2), stride = (2,2)),
						  Convolution2D(output_channels = 122, kernel_size = (3,3), stride = (1,1)),
						  MaxPooling(pool_size = (2,2), stride = (2,2)),
						  Flatten(),
						  Dense(num_nodes = 1000),
						  Dense(num_nodes = 10, output_layer = True)],
						  10)

	nn.feed_forward()
	#a = np.random.rand(28,28)
	#print(measure.block_reduce(a, (2,2), np.max).shape)
	# result = np.array([convolve2d(img, kernel[i,:,:], mode = 'valid') for i in range(kernel.shape[0])])
	# print(result.shape)
	#print(lambda x:)

	# print(img)
	# print(measure.block_reduce(img, (3,3), np.max))
	#print(convolve2d(img, kernel[0,:,:], mode = 'valid'))
	#print(img[0,:,:])

	# a = np.array([[[[1,1,1],
	# 			   [1,1,1]],
	# 			   [[2,2,2],
	# 			   [2,2,2]],
	# 			   [[3,3,3],
	# 			   [3,3,3]]],

	# 			   [[[4,4,4],
	# 			   [4,4,4]],
	# 			   [[5,5,5],
	# 			   [5,5,5]],
	# 			   [[6,6,6],
	# 			   [6,6,6]]]])
	# print(a.shape)
	# print(a.reshape(2,-1))

	



	
