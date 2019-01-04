###########################################################################
#
# K-Means Clustering Implementation Object
# **Initial K-means exploration and analysis done here**
#
# Author: Sam Showalter
# Date: November 17, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Systems based information
import os
import sys
from pathlib import Path

#Logistic and system-based imports
import datetime as dt 
import pandas as pd

#Package specific imports
from Customer_Segmentation import UI_KMeans_Orch

#Web framework information
from flask import Flask, flash, redirect, render_template, request, session, abort

###########################################################################
# Full test execution orchestrators
###########################################################################

app = Flask(__name__)
 
@app.route("/")
def run_app():
	print("HIHIHI")
	return render_template('index.html')

@app.route('/', methods=['POST'])
def run_customer_segmentation():
	
	#Silhouette analysis conditional code
	silhouette_boolean = False
	silhouette_range_start = 0
	silhouette_range_end = 0

	#Silhouette analysis optional code
	if request.form['silhouette_boolean'] == "True":
		silhouette_boolean = True
		silhouette_range_start = int(request.form["silhouette_range_start"])
		silhouette_range_end = int(request.form["silhouette_range_end"])

	UI_KMeans_Orch(
					#Training Data
				   	pd.read_csv(str(Path.home()) + "\\Desktop\\" + request.form['train_data']),

				   	#Original Data
				   	pd.read_csv(str(Path.home()) + "\\Desktop\\" + request.form['original_data']),

				   	#Elbow chart cluster range
				   	range(int(request.form["cluster_range_start"]), int(request.form["cluster_range_end"]) + 1),

				   	#Silhouette analysis information
				   	silhouette_analysis = silhouette_boolean,
				   	silhouette_cluster_range = range(silhouette_range_start,silhouette_range_end + 1),

				   	#Test and visualization names kept the same for now
				   	viz_name = request.form["test_name"],
				   	test_name = request.form["test_name"])

	return "Customer Segmentation Has Finished!"

    
 
if __name__ == "__main__":
	os.system("start chrome http://localhost:5000")
	app.run(host = '127.0.0.1', port = 5000)