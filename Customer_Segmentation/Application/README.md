# CustomerSegmentation
## Author: Samuel Showalter
### West Monroe Partners
### Date: September 14, 2018

# Customer Segmentation Analytics Package.

This software package serves to streamline the analysis process behind customer segmentation. Customer segmentation and other clustering analyses are heavily dependent on having high quality, representative data rolled up to the relevant grain (customer, household, etc.). This package does not facilitate any of that process. 

Rather, the primary role of this software is to ensure that once data has been correctly prepared, clustering analysis can be quickly conducted and automatically logged. Clustering analysis for customer segmentation can be thought of in three parts:

1. Data collection, refinement, and organization/flattening
2. Clustering execution
3. Post-execution analysis and insights

This analytics package automates the second step in this process, allowing users the ability to quickly move into the final insights phase with all the post-execution data in-hand. The value a package of this structure adds is its flexibility to accommodate many different types of clustering tests and parameters. Additionally, the logging functionality automatically saves all relevant information in an organized, user-friendly manner. While the current implementation of this code only supports K-Means clustering analysis, future releases may provide entirely new approaches of unsupervised learing that can be completed within this same package. 

# Getting Started

## Dependencies:

All dependencies listed have an easy install method through either `pip` or `conda`.

 - numpy
 - pandas
 - matplotlib
 - seaborn
 - datetime
 - sci-kit learn

## Classes

Several classes coordinate the execution of customer segmentation.

 - Customer_Segmentation
 - Preprocessing
 - Logger
 - Seg_Viz
 - Seg_Method
     + K_Means
 
### Package Class UML Diagram

<p align="center"><img src="https://image.ibb.co/nJxEZp/Customer_Segmentation.png" alt="Customer_Segmentation" width="800" height="800"/>

The class diagram for these objects is outlined above. `CustomerSegmentation` objects represent an orchestration of all other objects. Essentially, if someone wants to conduct a top-to-bottom segmentation analysis with this package, they must create a `CustomerSegmentation` object with the desired parameters. CustomerSegmentation objects take a `SegMethod` implementation object (at the moment, only KMeans() is available) and optional `Log` objects as its inputs. It can also optionally be given a dataset as a pandas dataframe, a test name, and a random state seed. Each of these classes and their attributes is discussed below.

# Features
- Preprocessing of training data to facilitate optimal clustering
- Iterative, replicable clustering analysis with a variety of implementations
    + Eventually,  `KMeans` will not be the only potential method for analysis
- Full logging of all execution metadata at different levels of granularity. These levels are listed below.
    + Master Log: Low granularity; overview of entire test
    + Preprocess Log: High granularity; Preprocess actions documented step-by-step
    + Method Log; High granularity; Segmentation method implemented step-by-step

Below, each package is explained with a high level of detail such that their parameters and interaction with other packages is clearly understood.

### `Customer_Segmentation`.CustomerSegmentation(
                                                         method,
                                                         data = pd.DataFrame(),
                                                         log = None,
                                                         test_name = "",
                                                         random_state = 42):

Initializes entire customer segmentation test. The user can then reference both preprocessing and segmentation methods from this object. Conversely, after the user becomes familiar with the data, they can create an run an orchestration package that completes all requested preprocessing and segmentation steps. A K-Means clustering orchestrator is already created and can be tuned for different use cases.

#### Parameters:
- **method**: Clustering implementation class object. Available classes are {`KMeans`}. 
- **data**: Input data as a Pandas Dataframe. Preferably with train data and class label.
- **log**: Log object from class `logger`. Used here as a Master Log.
- **test_name**: Name given to the test. Propagated through logging files and folders.
- **random_state**: Random seed for number generation. Default set as 42.

### `Preprocessing`.Preprocess(
                                    CustomerSeg):

Preprocessing package. Applies transformations to training data to make clustering easier. An example of a process would be Principal Component Analysis (PCA). Other attributes are tied to the preprocess class as functions are called. Though not an input, a PreprocessLog object stores all the data from the execution for later viewing.

#### Parameters:
- **CustomerSeg**: Customer segmentation test parent object.

### `Seg_Method`.SegMethod(
                                CustomerSeg,
                                Method):

Package that conducts clustering analysis. May utilize other unsupervised learning tests in the future. Though not an input, a MethodLog object stores all the data from the execution for later viewing.

#### Parameters:
- **CustomerSeg**: MeasSim object that contains all measurement data in the form of measurement frequency arrays.
- **Method**: Clustering implementation object (only `KMeans` available currently). 

### `K_Means`.KMeans():

K-means clustering implementation object that relies heavily on the sci-kit learn unsupervised learning code.

#### Parameters:
- There are no parameters because implementation classes feed up to the SegMethod parent and do not store any information within the class itself. It is a child class object because it needs access to `SegMethod` attributes.

### `Logger`.Log(
                    master_log_name, 
                    preprocess_log_name, 
                    method_log_name,
                    directory = ""):

Logger object that stores and organizes all data from an execution. One instantiation of a Log can be a Master Log, Preprocess Log, or Method Log, but not more than one instantiation at once. All logs are organized using transferrence of file names and folders named accordingly. Log filenames are are all suffixed with the timestamp `YYYY-MM-DD_hh.mm.ss.fff` where 'f' represents fractions of a second. Master logs do not leverage the fractions of a second, and truncate them from the name.

#### Parameters:
- **master_log_name**: Master Log name. Time stamps and other features are added to these names to ensure the file is unique and does not override other data.
- **preprocess_log_name**: Preprocess log name. Saved in a folder with a name corresponding to the relevant Master Log and test number.
- **method_log_name**: Method log name. Saved in a folder with a name corresponding to the relevant Master Log and test number. 
- **directory**: Folder where all logging output will be stored. It is highly recommended that you provide a path for this. Provide escape characters when necessary.
             
### `SegViz`

SegViz is a static store of visualization functions. It is not an object itself, nor is it owned by any parent objects. All objects in this execution have access to seg_viz methods, and utilize them any time an analysis step can be supplemented with a visualization.

# Logging Outputs and Organization

A single CustomerSegmentation test creates one folder and one flat file as part of its execution. The flat file represents the Master_Log summary data, and is named as such. The folder has the same name as the Master_Log, and contains list of folders, one for each test.

In a test folder with name schema `Test_#` where `#` is the actual test number, the Preprocess_Log, Method_Log, and saved visualizations are stored for that test. A visual break-out of the folder structure for a Master_Log with two tests is shown below. Test 2 has a break-out of its contents, which have the same structure as the contents of Test 1.

<img src="https://image.ibb.co/m6OkPp/Cause_Effect_Diagram.png" alt="Cause_Effect_Diagram" width="800" height="800"/>

# Citations:
1. [Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

# Example

Below is a demonstration on how one might run a Customer Segmentation experiment using this package. 

```
        #Import sample datasets
        from sklearn import datasets

        #Load iris dataset
        iris = datasets.load_iris()

        #Create Master Log
         MasterLog = Log("Master-Log", "Preprocess-Log", "SegMethod-Log")

        #Run K-means Orchestration package
        K_Means_Orch(
                     # Reference to the master log
                     MasterLog,

                     #Training data
                     iris.data,

                     #Class labels
                     iris.target,

                     #Elbow chart cluster range
                     range(2,12),

                     #Silhouette analysis cluster range
                     range(3,6),

                     #Column names for training data
                     train_col_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],

                     #Visualization information (do we want pictures? See them at runtime?)
                     viz = True,
                     show = False,
                     viz_name = "Iris_Data",

                     #Test Name
                     test_name = "Iris_Data")
```

This test runs the K-Means clustering orchestration package. With the training data and class labels, Principal Component analysis is run and a cumulative variance plot is used to determine the optimal number of components. Next, K_means clustering is run for a range of cluster sizes (3 clusters, 4 clusters, etc.) in the form on an elbow plot analysis. To further analyze the elbow plot, silhouette analysis was completed on a subset of the elbow plot analysis range. Finally, all of this data is stored in `Test_1` in the relevant Master_Log folder.

# Output Demo

Here is an output demo of what happens when the orchestration code above is run. The gif shows the files created in the specified output directory. This can be set to any directory on your computer.

![](https://image.ibb.co/dREO6e/Customer_Seg_Demo.gif)

# Installation
You may obtain the software from West Monroe Partners' Git repository. If you are having trouble gaining access, please contact [Sam Showalter](mailto:sshowalter@wmp.com).

The software was written for Python 3. To create a Python 3 environment,
we recommend using [Anaconda](https://www.anaconda.com/download/). 

For more information and instructions see [Conda's User Guide](https://conda.io/docs/user-guide/tasks/manage-environments.html)

All three scripts mentioned above can be run from the terminal, a Python Shell, or a developement environment like Spyder 
(included with Anaconda).

# Support
If you are have questions or issues, please contact [Sam Showalter](mailto:sshowalter@wmp.com).