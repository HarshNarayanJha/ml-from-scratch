# Implementing ML Algorithms from Scratch

This repo contains the implementation for a few basic ML algorithms from scratch in python

### 1. K-Nearest-Neighbours (KNN)

Given a data point

- Calculate its distance from all other data points (slow)
- Get the closest K points (hyperparameter, user-determined)
- _Regression_: Get the average of their values
- _Classification_: Get the label with majority vvote

Find the implementation in the file `knn.py`
