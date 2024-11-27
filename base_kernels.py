import numpy as np
from scipy.spatial.distance import cdist


### define the kernels
class BaseKernels:
    def __init__(self, locations):
        self.locations = locations
        self.N = len(locations)

    ## linear kernels
    
    # linear kernel over x,y, i.e. similarity as a function of the distance from the origin (0,0)
    def linear(self, c=1):
        dists = np.sqrt(self.locations[:, 0]**2 + self.locations[:, 1]**2)
        K = np.outer(dists, dists) + c
        return K

    # linear kernel over x-distance (0) or y-distance (1), i.e. similarity as a function of the distance from (0,:) or (:,0), where the basis vectors are determined by the angle theta (in radians)    
    def linear_1D(self, dim=0, theta=0, scale=1, c=1):

        # define basis vectors
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_locations = np.dot(self.locations, rotation)

        # calculate distances
        dists = rotated_locations[:, dim] * scale**2
        K = np.outer(dists, dists) + c
        return K



    ## RBFs 

    # RBF kernel over x,y (i.e. Euclidean distance)
    def rbf(self, sigma_f=1, length_scale=0.5):
        dists = cdist(self.locations, self.locations, metric='euclidean')
        K = sigma_f**2 * np.exp(-0.5 * (dists / length_scale)**2)
        return K

    # RBF kernel over just x-distance or y-distance
    def rbf_1D(self, dim=0, theta=0, sigma_f=1, length_scale=0.5):

        # define basis vectors
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_locations = np.dot(self.locations, rotation)

        # calculate distances
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = sigma_f**2 * np.exp(-0.5 * (dists / length_scale)**2)
        return K


    ## periodic kernel
    def periodic(self, dim=0, sigma_f=1, period=1, periodic_length_scale=0.5, periodic_theta=np.pi/3):
        rotation = np.array([[np.cos(periodic_theta), -np.sin(periodic_theta)], [np.sin(periodic_theta), np.cos(periodic_theta)]])
        rotated_locations = np.dot(self.locations, rotation)
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = sigma_f**2 * np.exp(-2 * np.sin(np.pi * dists / period)**2 / periodic_length_scale**2)
        return K
