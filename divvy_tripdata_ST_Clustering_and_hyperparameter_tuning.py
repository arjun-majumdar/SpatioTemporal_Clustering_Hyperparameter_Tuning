

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from scipy.spatial.distance import pdist, squareform
# from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import scale, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Read in CSV file containing dataset-
data = pd.read_csv("202004-divvy-tripdata.csv")

# Get dimension/shape of file-
data.shape
# (84776, 13)


# Check for missing values in dataset-
data.isnull().values.any()
# True

data.isnull().sum().sum()
# 396

data.isnull().sum()
'''
ride_id                0
rideable_type          0
started_at             0
ended_at               0
start_station_name     0
start_station_id       0
end_station_name      99
end_station_id        99
start_lat              0
start_lng              0
end_lat               99
end_lng               99
member_casual          0
dtype: int64
'''


# To get summary about Pandas DataFrame-
data.info()


# To get value distribution for 'rideable_type' attribute-
data['rideable_type'].value_counts()
'''
docked_bike    84776
Name: rideable_type, dtype: int64
'''

# Delete the following attribute-
data.drop('rideable_type', axis = 1, inplace = True)
data.drop('ride_id', axis = 1, inplace = True)
data.drop('start_station_name', axis = 1, inplace = True)
data.drop('end_station_name', axis = 1, inplace = True)

# Reset index of dataset-
data.reset_index(inplace = True, drop = True)


# Convert 'date' attribute to Pandas datetime format-
data['started_at'] = pd.to_datetime(data['started_at'])
data['ended_at'] = pd.to_datetime(data['ended_at'])


# Sort dataset according to 'date' attribute in ascending order-
data.sort_values(by = 'started_at', ascending = True, inplace = True)

# Time difference for first record-
data.loc[0, 'ended_at'] - data.loc[0, 'started_at']
# Timedelta('0 days 00:26:49')


# Get a small slice of data-
data_slice = data.loc[:499, ['started_at', 'start_lat', 'end_lat']]


# Convert 'date' attribute to a new column having unique int values
# instead of datetime values-
data_slice['date_int'] = data_slice['started_at'].rank(method='dense')

# Drop the following attribute-
data_slice.drop('started_at', axis = 1, inplace = True)




class ST_KMeans(BaseEstimator, TransformerMixin):
	"""
	Note that K-means clustering algorithm is designed for Euclidean distances.
	It may stop converging with other distances, when the mean is no longer a
	best estimation for the cluster 'center'.

	The 'mean' minimizes squared differences (or, squared Euclidean distance).
	If you want a different distance function, you need to replace the mean with
	an appropriate center estimation.


	Parameters:

	k:	number of clusters
    
	eps1 : float, default=0.5
		The spatial density threshold (maximum spatial distance) between 
		two points to be considered related.

	eps2 : float, default=10
		The temporal threshold (maximum temporal distance) between two 
		points to be considered related.

	metric : string default='euclidean'
		The used distance metric - more options are
		‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
		‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
		‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
		‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
	
	n_jobs : int or None, default=-1
		The number of processes to start; -1 means use all processors (BE AWARE)


	Attributes:
    
	labels : array, shape = [n_samples]
		Cluster labels for the data - noise is defined as -1
    """

	def __init__(self, k, eps1 = 0.5, eps2 = 10, metric = 'euclidean', n_jobs = 1):
		self.k = k
		self.eps1 = eps1
		self.eps2 = eps2
		# self.min_samples = min_samples
		self.metric = metric
		self.n_jobs = n_jobs


	def fit(self, X, Y = None):
		"""
		Apply the ST K-Means algorithm 
        
		X : 2D numpy array. The first attribute of the array should be time attribute
			as float. The following positions in the array are treated as spatial
			coordinates.
			The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            
			For example 2D dataset:
			array([[0,0.45,0.43],
			[0,0.54,0.34],...])


		Returns:

		self
		"""
        
		# check if input is correct
		X = check_array(X)

		# type(X)
		# numpy.ndarray

		# Check arguments for DBSCAN algo-
		if not self.eps1 > 0.0 or not self.eps2 > 0.0:
			raise ValueError('eps1, eps2, minPts must be positive')

		# Get dimensions of 'X'-
		# n - number of rows
		# m - number of attributes/columns-
		n, m = X.shape


		# Compute sqaured form Euclidean Distance Matrix for 'time' and spatial attributes-
		time_dist = squareform(pdist(X[:, 0].reshape(n, 1), metric = self.metric))
		euc_dist = squareform(pdist(X[:, 1:], metric = self.metric))

		'''
		Filter the euclidean distance matrix using time distance matrix. The code snippet gets all the
		indices of the 'time_dist' matrix in which the time distance is smaller than 'eps2'.
		Afterward, for the same indices in the euclidean distance matrix the 'eps1' is doubled which results
		in the fact that the indices are not considered during clustering - as they are bigger than 'eps1'.
		'''
		# filter 'euc_dist' matrix using 'time_dist' matrix-
		self.dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)


		# Initialize K-Means clustering model-
		self.kmeans_clust_model = KMeans(
			n_clusters = self.k, init = 'k-means++',
			n_init = 10, max_iter = 300,
			precompute_distances = 'auto', algorithm = 'auto')

		# Train model-
		self.kmeans_clust_model.fit(self.dist)


		self.labels = self.kmeans_clust_model.labels_
		self.X_transformed = self.kmeans_clust_model.fit_transform(self.dist)

		return self


	def transform(self, X):
		# print("\nX.shape = {0}\n".format(X.shape))
		# pass
		# return self.kmeans_clust_model.fit_transform(self.dist)
		# return self
		# return self.X_transformed
		
		# if type(X) is np.ndarray:
		if not isinstance(X, np.ndarray):
			# Convert to numpy array-
			X = X.values

		# Get dimensions of 'X'-
		# n - number of rows
		# m - number of attributes/columns-
		n, m = X.shape


		# Compute sqaured form Euclidean Distance Matrix for 'time' and spatial attributes-
		time_dist = squareform(pdist(X[:, 0].reshape(n, 1), metric = self.metric))
		euc_dist = squareform(pdist(X[:, 1:], metric = self.metric))

		# filter 'euc_dist' matrix using 'time_dist' matrix-
		dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

		return self.kmeans_clust_model.transform(dist)
		
		# return self.kmeans_clust_model.transform(X)
		




# Initialize ST-K-Means object-
st_kmeans_algo = ST_KMeans(
	k = 5, eps1=0.6,
	eps2=9, metric='euclidean',
	n_jobs=1
	)

Y = np.zeros(shape = (500,))

data_slice = data_slice.values

# Train on a chunk of dataset-
st_kmeans_algo.fit(data_slice, Y)

# Get clustered data points labels-
kmeans_labels = st_kmeans_algo.labels


# Get labels for points clustered using trained model-
# kmeans_transformed = st_kmeans_algo.X_transformed
kmeans_transformed = st_kmeans_algo.transform(data_slice)

kmeans_transformed.shape, kmeans_labels.shape
# ((500, 5), (500,))




dtc = DecisionTreeClassifier()

dtc.fit(kmeans_transformed, kmeans_labels)

y_pred = dtc.predict(kmeans_transformed)

# Get model performance metrics-
accuracy = accuracy_score(kmeans_labels, y_pred)
precision = precision_score(kmeans_labels, y_pred, average='macro')
recall = recall_score(kmeans_labels, y_pred, average='macro')

print("\nDT model metrics are:")
print("accuracy = {0:.4f}, precision = {1:.4f} & recall = {2:.4f}\n".format(
	accuracy, precision, recall
	))
# DT model metrics are:
# accuracy = 1.0000, precision = 1.0000 & recall = 1.0000




# Hyper-parameter Tuning:

# Define steps of pipeline-
pipeline_steps = [
	('st_kmeans_algo' ,ST_KMeans(k = 5, eps1=0.6, eps2=9, metric='euclidean', n_jobs=1)),
	('dtc', DecisionTreeClassifier())
	]

# Instantiate a pipeline-
pipeline = Pipeline(pipeline_steps)

kmeans_transformed.shape, kmeans_labels.shape
# ((500, 5), (500,))

# Train pipeline-
pipeline.fit(kmeans_transformed, kmeans_labels)

pipeline.predict(kmeans_transformed).shape
# (500,)


# Specify parameters to be hyper-parameter tuned-
params = [
	{
		'st_kmeans_algo__k': [3, 5, 7],
		# 'st_kmeans_algo__metric': ['euclidean', 'cityblock'],
		'st_kmeans_algo__eps1': [0.6, 0.8, 0.9],
		'st_kmeans_algo__eps2': [9, 11],
	}
	]


# Initialize a RandomizedSearch CV object-
random_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=10, cv = 2, verbose=True)

# Train randomized search CV on data from above-
random_cv.fit(kmeans_transformed, kmeans_labels)

print("\nBest estimator using RandomizedSearchCV found:\n{0}\n".format(random_cv.best_estimator_))
'''
Best estimator using RandomizedSearchCV found:
Pipeline(memory=None,
         steps=[('st_kmeans_algo',
                 ST_KMeans(eps1=0.8, eps2=9, k=5, metric='euclidean',
                           n_jobs=1)),
                ('dtc',
                 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        presort='deprecated', random_state=None,
                                        splitter='best'))],
         verbose=False)

'''

print("\nBest score achieved using RandomizedSearchCV = {0:.4f}%\n".format(random_cv.best_score_ *100))
# Best score achieved using RandomizedSearchCV = 39.4000%

print("\nBest parameters found using RandomizedSearchCV:\n{0}\n".format(random_cv.best_params_))
'''
Best parameters found using RandomizedSearchCV:
{'st_kmeans_algo__k': 5, 'st_kmeans_algo__eps2': 9, 'st_kmeans_algo__eps1': 0.8}
'''



# Specify parameters to be hyper-parameter tuned found using RandomizedSearchCV from above-
params_grid = [
	{
		'st_kmeans_algo__k': [4, 5, 6],
		# 'st_kmeans_algo__metric': ['euclidean', 'cityblock'],
		'st_kmeans_algo__eps1': [0.7, 0.8, 0.9],
		'st_kmeans_algo__eps2': [8, 9, 10],
	}
	]


# Initialize GridSearchCV object-
grid_cv = GridSearchCV(estimator = pipeline, param_grid = params_grid, cv = 2)

# Train GridSearch on computed data from above-
grid_cv.fit(kmeans_transformed, kmeans_labels)


print("\nBest estimator found using GridSearchCV:\n{0}\n".format(grid_cv.best_estimator_))
'''
Best estimator found using GridSearchCV:
Pipeline(memory=None,
         steps=[('st_kmeans_algo',
                 ST_KMeans(eps1=0.7, eps2=9, k=4, metric='euclidean',
                           n_jobs=1)),
                ('dtc',
                 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        presort='deprecated', random_state=None,
                                        splitter='best'))],
         verbose=False)

'''


print("\nBest score achieved using GridSearchCV= {0:.4f}%\n".format(grid_cv.best_score_ *100))
# Best score achieved using GridSearchCV= 39.4000%

print("\nBest parameters found using GridSearchCV:\n{0}\n".format(grid_cv.best_params_))
'''
Best parameters found using GridSearchCV:
{'st_kmeans_algo__eps1': 0.7, 'st_kmeans_algo__eps2': 9, 'st_kmeans_algo__k': 4}
'''




class ST_DBSCAN(BaseEstimator, TransformerMixin):
	"""
    
	Parameters:
    
	eps1 : float, default=0.5
		The spatial density threshold (maximum spatial distance) between 
		two points to be considered related.

	eps2 : float, default=10
		The temporal threshold (maximum temporal distance) between two 
		points to be considered related.

	min_samples : int, default=5
		The number of samples required for a core point.

	metric : string default='euclidean'
		The used distance metric - more options are
		‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
		‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
		‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
		‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
	
	n_jobs : int or None, default=-1
		The number of processes to start; -1 means use all processors (BE AWARE!)


	Attributes:
    
	labels : array, shape = [n_samples]
		Cluster labels for the data - noise is defined as -1
    """

	def __init__(self, eps1 = 0.5, eps2 = 10, min_samples = 5, metric = 'euclidean', n_jobs = 1):
		self.eps1 = eps1
		self.eps2 = eps2
		self.min_samples = min_samples
		self.metric = metric
		self.n_jobs = n_jobs


	def fit(self, X, Y = None):
		"""
		Apply the ST DBSCAN algorithm 
        
		X : 2D numpy array. The first attribute of the array should be time attribute
			as float. The following positions in the array are treated as spatial
			coordinates.
			The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            
			For example 2D dataset:
			array([[0,0.45,0.43],
			[0,0.54,0.34],...])


		Returns:

		self
		"""
        
		# check if input is correct
		X = check_array(X)

		# type(X)
		# numpy.ndarray

		# print("\neps1 = {0:2f}, eps2 = {1:.2f} & min_samples = {2}\n".format(self.eps1, self.eps2, self.min_samples))

		# Check arguments for DBSCAN algo-
		if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
			raise ValueError('eps1, eps2, minPts must be positive')

		# Get dimensions of 'X'-
		# n - number of rows
		# m - number of attributes/columns-
		n, m = X.shape


		# Compute sqaured form Euclidean Distance Matrix for 'time' and spatial attributes-
		time_dist = squareform(pdist(X[:, 0].reshape(n, 1), metric=self.metric))
		euc_dist = squareform(pdist(X[:, 1:], metric=self.metric))

		'''
		Filter the euclidean distance matrix using time distance matrix. The code snippet gets all the
		indices of the 'time_dist' matrix in which the time distance is smaller than 'eps2'.
		Afterward, for the same indices in the euclidean distance matrix the 'eps1' is doubled which results
		in the fact that the indices are not considered during clustering - as they are bigger than 'eps1'.
		'''
		# filter 'euc_dist' matrix using 'time_dist' matrix-
		# dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)
		self.dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)


		# Initialize DBSCAN model-
		self.dbs = DBSCAN(
			eps=self.eps1, min_samples = self.min_samples,
			metric='precomputed'
			)


		# Train model-
		self.dbs.fit(self.dist)


		self.labels = self.dbs.labels_
		# self.X_transformed = self.dbs.fit_transform(self.dist)

		return self


	def transform(self, X):
		# if type(X) is np.ndarray:
		if not isinstance(X, np.ndarray):
			# Convert to numpy array-
			X = X.values

		# Get dimensions of 'X'-
		# n - number of rows
		# m - number of attributes/columns-
		n, m = X.shape


		# Compute sqaured form Euclidean Distance Matrix for 'time' and spatial attributes-
		time_dist = squareform(pdist(X[:, 0].reshape(n, 1), metric = self.metric))
		euc_dist = squareform(pdist(X[:, 1:], metric = self.metric))

		# filter 'euc_dist' matrix using 'time_dist' matrix-
		dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)


		if len(set(self.labels)) == 1:
			k = 1
		else:
			k = len(set(self.labels)) - 1

		# Initialize K-Means clustering model-
		self.kmeans_clust_model = KMeans(
			n_clusters = k, init = 'k-means++',
			n_init = 10, max_iter = 300,
			precompute_distances = 'auto', algorithm = 'auto')

		# Train model-
		self.kmeans_clust_model.fit(dist)

		# return self.kmeans_clust_model
		return self.kmeans_clust_model.transform(dist)




# Initialize ST-DBSCAN object-
st_dbscan = ST_DBSCAN(
	eps1 = 0.6, eps2 = 9,
	min_samples = 5,
	metric='euclidean',
	n_jobs = 1
)


# Train on a chunk of dataset-
st_dbscan.fit(data_slice, Y)

# Get clustered data points labels-
dbscan_labels = st_dbscan.labels


# Get labels for points clustered using trained model-
dbscan_kmeans_transformed = st_dbscan.transform(data_slice)


dbscan_kmeans_transformed.shape, dbscan_labels.shape
# ((500, 1), (500,))


dtc = DecisionTreeClassifier()

dtc.fit(dbscan_kmeans_transformed, dbscan_labels)

y_pred = dtc.predict(dbscan_kmeans_transformed)

# Get model performance metrics-
accuracy = accuracy_score(dbscan_labels, y_pred)
precision = precision_score(dbscan_labels, y_pred, average='macro')
recall = recall_score(dbscan_labels, y_pred, average='macro')

print("\nDT model metrics are:")
print("accuracy = {0:.4f}, precision = {1:.4f} & recall = {2:.4f}\n".format(
	accuracy, precision, recall
	))
# DT model metrics are:
# accuracy = 1.0000, precision = 1.0000 & recall = 1.0000


# Hyper-parameter Tuning:

# Define steps of pipeline-
pipeline_steps_dbscan = [
	('st_dbscan_algo' ,ST_DBSCAN(eps1=0.6, eps2=9, min_samples = 5, metric='euclidean', n_jobs=1)),
	('dtc', DecisionTreeClassifier())
	]

# Instantiate a pipeline-
pipeline_dbscan = Pipeline(pipeline_steps_dbscan)

# Train pipeline-
pipeline_dbscan.fit(dbscan_kmeans_transformed, dbscan_labels)

pipeline_dbscan.predict(dbscan_kmeans_transformed).shape
# (500,)


# Specify parameters to be hyper-parameter tuned-
params_rs_dbscan = [
	{
		# 'st_kmeans_algo__metric': ['euclidean', 'cityblock'],
		'st_dbscan_algo__eps1': [0.2, 0.4, 0.6, 0.8, 0.99],
		'st_dbscan_algo__eps2': [2, 4, 6, 8, 11],
		'st_dbscan_algo__min_samples': [2, 4, 6, 8, 10]
	}
	]




# Initialize a RandomizedSearch CV object-
random_cv_db = RandomizedSearchCV(estimator=pipeline_dbscan, param_distributions=params_rs_dbscan, n_iter=10, cv = 2, verbose=True)

# Train randomized search CV on data from above-
random_cv_db.fit(dbscan_kmeans_transformed, dbscan_labels)

print("\nBest DBSCAN estimator using RandomizedSearchCV found:\n{0}\n".format(random_cv_db.best_estimator_))
'''
Best DBSCAN estimator using RandomizedSearchCV found:
Pipeline(memory=None,
         steps=[('st_dbscan_algo',
                 ST_DBSCAN(eps1=0.6, eps2=6, metric='euclidean', min_samples=6,
                           n_jobs=1)),
                ('dtc',
                 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        presort='deprecated', random_state=None,
                                        splitter='best'))],
         verbose=False)

'''

print("\nBest score achieved using DBSCAN with RandomizedSearchCV = {0:.4f}%\n".format(random_cv_db.best_score_ *100))
# Best score achieved using DBSCAN with RandomizedSearchCV = 100.0000%

print("\nBest parameters found using DBSCAN with RandomizedSearchCV:\n{0}\n".format(random_cv_db.best_params_))
'''
Best parameters found using DBSCAN with RandomizedSearchCV:
{'st_dbscan_algo__min_samples': 6, 'st_dbscan_algo__eps2': 6, 'st_dbscan_algo__eps1': 0.6}
'''



def plot(data, labels):

	colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

	for i in range(-1, len(set(labels))):
		if i == -1:
			col = [0, 0, 0, 1]
		else:
			col = colors[i % len(colors)]
        
		clust = data[np.where(labels == i)]
		plt.scatter(clust[:,0], clust[:,1], c=[col], s=1)

	plt.show()

	return None


# plot(data_slice, kmeans_labels)
# plot(data_slice, dbscan_labels)


