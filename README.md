# SpatioTemporal_Clustering_Hyperparameter_Tuning
Hyperparameter Tuning for some common Spatio-Temporal Clustering algorithms

Most of the clustering algorithms don't have hyperparameter tuning to find the 'optimal' hyperparameters in order to fine-tune according to a given dataset, this repository aims to address this shortcoming by implementing this for the following clustering algorithms:

1. _K-Means_
1. _DBSCAN_
1. _Gaussian Mixture Models_
1. _Agglomerative_
1. _HDBSCAN_


The hyperparameter tuning first uses _RandomizedSearchCV_ followed by _GridSearchCV_ to hopefully converge on the appropriate hyperparameter set.
