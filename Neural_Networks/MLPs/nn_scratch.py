###########################################################################
#
# Neural Network Implementation from Scratch
# -- Built for a WMP KTS --
#
# Author: Sam Showalter
# Date: October 11, 2018
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

#Dataset related imports
import mnist
from sklearn import datasets

###########################################################################
# Data formatting and restructuring for analysis 
###########################################################################

x_train_mnist = mnist.train_images()
x_test_mnist = pd.get_dummies(pd.Series(mnist.train_labels())).values

y_train_mnist = mnist.test_images()
y_train_mnist.resize(10000,28*28)
y_train_mnist = y_train_mnist / 255.0

y_test_mnist = pd.get_dummies(pd.Series(mnist.test_labels())).values


#print(pd.Series(x_test_mnist.shape))
x_train_mnist.resize(60000,28*28)
x_train_mnist = x_train_mnist / 255.0


x_train_mnist_1 = x_train_mnist[:10000]
y_train_mnist_1 = x_test_mnist[:10000]

x_test_mnist_1 = x_train_mnist[10000:]
y_test_mnist_1 = x_test_mnist[10000:]

print(x_test_mnist_1.shape)
print(y_test_mnist_1.shape)



iris = datasets.load_iris()
train = pd.DataFrame(iris.data) 
test = pd.Series(iris.target)


test =(pd.get_dummies(pd.Series(test)))


x_train, x_test,y_train, y_test = train_test_split(train, test, test_size = 0.33, random_state = 42)

# print(pd.get_dummies(y_train))

# y_train = np.array((pd.get_dummies(pd.Series(y_train))).values)


# print(x_train)
# print(y_train)


###########################################################################
# Class and constructor
###########################################################################


class NN_Scratch():
	"""
	This class implements single and multi-layer neural networks (perceptrons)
	from scratch. Primarily, this class is to be used for teaching purposes, though
	with a small amount of tweaking it could be applied to a client project. Networks
	are optimized with backpropagation via batch gradient descent. Other methods of 
	backpropagation will be implemented in the future.

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
					num_inputs, 
					num_hidden_layers, 
					hidden_layer_nodelist, 
					num_outputs,
					activation_function = "tanh",
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

				num_inputs:					Number of input features into the network
				num_hidden_layers:			Number of hidden layers in the network (must be at least 1)
				hidden_layer_nodelist:		The number of nodes to have in each hidden layer
											Number of nodes is sequential with the layers
				num_outputs:				Number of class labels that can be predicted

					### Activation and backpropagation attributes ###

				activation_function:		Activation function keyword. Corresponds to key in activation_dict
				epsilon:					Learning rate for gradient descent
				reg_lambda:					Regularization strength

		"""

		#Initialize neural network
		self.directory = directory
		self.network_name = network_name
		self.network = {}

		#Activation dictionary; stores all activation functions
		self.activation_dict = {"tanh": self.tanh,
								"derivtanh": self.derivTanh}

		#Set train data and output labels
		self.full_train_data = train_data
		self.full_labels = labels

		#Initialize log_loss and log loss difference
		self.log_loss = np.Inf
		self.log_loss_dif = 0

		# Gradient descent parameters 
		self.epsilon = epsilon 			# learning rate for gradient descent
		self.reg_lambda = reg_lambda 	# regularization strength

		#Network dimension information
		self.num_inputs = num_inputs
		self.num_hidden_layers = num_hidden_layers
		self.hidden_layer_nodelist = hidden_layer_nodelist
		self.num_outputs = num_outputs

		#Activation function attribution from dictionary
		self.activate = self.activation_dict[activation_function]
		self.derivActivate = self.activation_dict["deriv" + activation_function]

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
		as random weights and zeros, respectively. 
		"""

		#Add first set of weights that connect to the input
		self.network["w0"] = np.random.rand(self.num_inputs, self.hidden_layer_nodelist[0]) #* self.epsilon
		self.network["b0"] = np.zeros((1, self.hidden_layer_nodelist[0]))

		#Add all intermediate hidden layers
		for i in range(self.num_hidden_layers - 1):
			self.network["w" + str(i + 1)] = np.random.rand(self.hidden_layer_nodelist[i], self.hidden_layer_nodelist[i+1]) #* self.epsilon
			self.network["b" + str(i + 1)] = np.zeros((1,self.hidden_layer_nodelist[i+1]))

		#Add weights that go to output layer
		self.network["w" + str(self.num_hidden_layers)] = np.random.rand(self.hidden_layer_nodelist[-1], self.num_outputs) #* self.epsilon
		self.network["b" + str(self.num_hidden_layers)] = np.zeros((1,self.num_outputs))
	

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
		for i in range(self.num_hidden_layers + 1):

			#If this is the input layer
			if (i == 0):
				throughput = np.matmul(data, self.network["w" + str(i)]) + self.network["b" + str(i)]

			#If this is a hidden layer
			else:
				throughput = np.matmul(self.network["a" + str(i - 1)], self.network["w" + str(i)]) + self.network["b" + str(i)]

			#Add the intermediate layer to the NN dictionary store
			self.network["z" + str(i)] = throughput

			#Ase activation function and add the results to NN
			if i < self.num_hidden_layers:
				throughput = self.activate(throughput)
				self.network["a" + str(i)] = throughput

		# Determine the output probabilities of each class with softmax
		self.probs = self.softmax(throughput)

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
		e_x = np.exp(throughput - np.max(throughput))
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

			input: Input to be altered by activation function for backprop
		"""
		return 1 - np.power(np.tanh(input_arr), 2)


###########################################################################
# Main method for testing
###########################################################################

if __name__ == '__main__':

	## -- Different versions and data tests to run -- ##
	# nn = NN_Scratch(y_train_mnist[:10000],y_test_mnist[:10000], 784, 1, [800], 10, 
	# 								network_name = "MNIST_data",
	# 								epsilon = 0.2,
	# 								network_filename = "MNIST_data_2018-10-15_14.40.14")
	#nn = NN_Scratch(x_train.values,y_train.values, 4,1,[10],3, network_name = "Iris_Data", epsilon = 0.05)
	#nn = NN_Scratch(x_train.values,y_train.values, 4,1,[10],3, network_name = "MNIST_Read_In", network_filename = "MNIST_data_2018-10-15_14.40.14")

	nn = NN_Scratch(x_train_mnist_1,y_train_mnist_1, 784, 1, [50], 10, 
									network_name = "MNIST_data",
	 								epsilon = 0.01)


	#nn = NN_Scratch(x_train.values,y_train.values, 784,1,[50],10, network_name = "MNIST_Read_In", network_filename = "MNIST_data_i784_h50_o10__2018-10-15_16.27.13")


	# epsilons, log_loss = nn.learning_rate_test(num_epochs = 500)
	# print(epsilons)
	# print(log_loss)
	# plt.plot(epsilons, log_loss)
	# plt.show()

	nn.train(	num_epochs = 5000,
				epoch_update_cadence = 100,
				verbose = True,
				update_learn_rate = True,
				gradient_method = "batch",
				batch_size = 100)

	# nn.predict(y_train_mnist, y_test_mnist)
	# print(nn.accuracy)

	nn.save_network()

	# nn.predict(x_test.values,y_test.values)
	# print("Final accuracy results are:   ")
	# print(nn.probs.argmax(axis = 1))
	# print(y_test.values.argmax(axis = 1))
	# print(nn.accuracy)


	nn.predict(x_test_mnist_1,y_test_mnist_1)
	print("Final accuracy results are:   ")
	print(nn.probs.argmax(axis = 1))
	print(y_test_mnist_1.argmax(axis = 1))
	print(nn.accuracy)

	# print(nn.network.keys())


	



	
