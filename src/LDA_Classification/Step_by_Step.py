"""Step by Step explaining how LDA works in 2 classes(Just use to explain)
"""

import numpy as np

from numpy import ndarray
from typing import Tuple

class StepByStepExplain():
    """StepByStep Class using for explaination
    """
    def __init__(self,X:ndarray,y:ndarray) -> None:
        """init function in step by step class

        Attributes:
            X (ndarray): A NumPy array of shape (n_samples, n_features), representing the feature data.
            y (ndarray): A NumPy array of shape (n_samples,), containing the target labels for each sample.
        Returns:
            None
        """
        self.X:ndarray = X
        self.y:ndarray = y

    def step_by_step(self)-> Tuple[ndarray,ndarray]:
        """Step by step calculate eigenvalues and eigenvector

        Returns:
            Tuple[ndarray,ndarray]: Tuple contains eigenvalue and eigen vector
        """
        X: ndarray = self.X  # Feature data: NumPy array of shape (n_samples, n_features)
        y: ndarray = self.y  # Labels: NumPy array of shape (n_samples,)

        # Calculate the mean vector for class 0
        u1: ndarray = np.mean(X[y == 0], axis=0)  # Mean vector for class 0, shape (n_features,)
        # Calculate the mean vector for class 1
        u2: ndarray = np.mean(X[y == 1], axis=0)  # Mean vector for class 1, shape (n_features,)

        # Extract data points belonging to class 0
        X_class_0: ndarray = X[y == 0]  # Data for class 0, shape (n_class_0_samples, n_features)
        # Initialize covariance matrix for class 0
        s1: ndarray = np.zeros((X.shape[1], X.shape[1]))  # Covariance matrix, shape (n_features, n_features)
        # Center data for class 0
        centered_X: ndarray = X_class_0 - u1  # Centered data, shape (n_class_0_samples, n_features)
        # Calculate covariance matrix of class 0
        s1 = np.dot(centered_X.T, centered_X) / (X_class_0.shape[1] - 1)  # Covariance matrix, shape (n_features, n_features)

        # Extract data points belonging to class 1
        X_class_1: ndarray = X[y == 1]  # Data for class 1, shape (n_class_1_samples, n_features)
        # Initialize covariance matrix for class 1
        s2: ndarray = np.zeros((X.shape[1], X.shape[1]))  # Covariance matrix, shape (n_features, n_features)
        # Center data for class 1
        centered_X = X_class_1 - u2  # Centered data, shape (n_class_1_samples, n_features)
        # Calculate covariance matrix of class 1
        s2 = np.dot(centered_X.T, centered_X) / (X_class_1.shape[1] - 1)  # Covariance matrix, shape (n_features, n_features)

        # Calculate within-class scatter matrix S_W
        S_w: ndarray = s1 + s2  # Within-class scatter matrix, shape (n_features, n_features)

        # Calculate between-class scatter matrix S_B
        U: ndarray = u1 - u2  # Difference of means, shape (n_features,)
        S_b: ndarray = np.dot(U.reshape(-1, 1), U.reshape(1, -1))  # Between-class scatter matrix, shape (n_features, n_features)

        # Calculate the inverse of S_W
        S_W_inv: ndarray = np.linalg.inv(S_w)  # Inverse matrix, shape (n_features, n_features)

        # Compute matrix for eigen decomposition
        M: ndarray = np.dot(S_W_inv, S_b)  # Matrix to decompose, shape (n_features, n_features)

        # Perform eigen decomposition
        eigenvalues: ndarray  # Eigenvalues of M, shape (n_features,)
        eigenvectors: ndarray  # Corresponding eigenvectors of M, shape (n_features, n_features)
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices: ndarray = np.argsort(eigenvalues)[::-1]  # Indices for sorting, shape (n_features,)
        sorted_eigenvalues: ndarray = eigenvalues[sorted_indices]  # Sorted eigenvalues, shape (n_features,)
        sorted_eigenvectors: ndarray = eigenvectors[:, sorted_indices]  # Sorted eigenvectors, shape (n_features, n_features)

        return sorted_eigenvalues, sorted_eigenvectors


    
    
    


    




