"""LDA for specified number of classes with better struc"""

import numpy as np

from numpy import ndarray


class LDA:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction and classification.

    Attributes:
        n_components (int): Number of components to keep.
        linear_discriminants (ndarray): Linear discriminants (eigenvectors) for the LDA transformation.
        class_means_trans (dict): Dictionary storing the mean of each class in the LDA-transformed space.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initializes the LDA model with the specified number of components.

        Args:
            n_components (int): Number of components to keep.
        """
        self.n_components: int = n_components
        # Eigenvectors of the LDA transformation
        self.linear_discriminants: ndarray | None = None
        # Mean of each class in the LDA-transformed space
        self.class_means_trans: dict[int, ndarray] = {}

    def fit(self, X: ndarray, y: ndarray) -> None:
        """
        Fits the LDA model to the data.

        Args:
            X (ndarray): Feature data of shape (n_samples, n_features).
            y (ndarray): Target labels of shape (n_samples,).

        Returns:
            None
        """
        n_features: int = X.shape[1]  # Number of features in the dataset
        # Unique class labels in the dataset
        class_labels: ndarray = np.unique(y)

        # Overall mean vector of the dataset
        mean_overall: ndarray = np.mean(X, axis=0)

        # Initialize within-class scatter matrix (S_W) and between-class scatter matrix (S_B)
        S_W: ndarray = np.zeros((n_features, n_features))
        S_B: ndarray = np.zeros((n_features, n_features))

        # Calculate S_W and S_B for each class
        for c in class_labels:
            X_c: ndarray = X[y == c]  # Data points belonging to class `c`
            mean_c: ndarray = np.mean(X_c, axis=0)  # Mean vector for class `c`
            n_c: int = X_c.shape[0]  # Number of samples in class `c`
            mean_diff: ndarray = (mean_c - mean_overall).reshape(n_features, 1)

            # Add to within-class scatter matrix
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            # Add to between-class scatter matrix
            S_B += n_c * mean_diff.dot(mean_diff.T)

        # Solve the generalized eigenvalue problem for S_W^-1 * S_B
        A: ndarray = np.linalg.inv(S_W).dot(S_B)
        eigenvalues: ndarray
        eigenvectors: ndarray
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T  # Transpose to align eigenvectors correctly

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idxs: ndarray = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Select the top `n_components` eigenvectors
        self.linear_discriminants: ndarray = eigenvectors[0 : self.n_components]

        # Transform the data and store means in the transformed space
        X_transformed: ndarray = self.tranform(X)
        for c in class_labels:
            X_c_trans: ndarray = X_transformed[y == c]
            self.class_means_trans[c] = np.mean(X_c_trans, axis=0)

    def tranform(self, X: ndarray) -> ndarray:
        """
        Transforms the data into the LDA space.

        Args:
            X (ndarray): Feature data of shape (n_samples, n_features).

        Returns:
            ndarray: Transformed data of shape (n_samples, n_components).
        """
        return np.dot(X, self.linear_discriminants.T)

    def predict(self, X: ndarray) -> ndarray:
        """
        Predicts the class labels for the input data.

        Args:
            X (ndarray): Feature data of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels of shape (n_samples,).
        """
        # Transform the input data into the LDA space
        transformed_X: ndarray = self.tranform(X)

        # Predict based on Euclidean distance to class means
        predictions: list[int] = []
        for sample in transformed_X:
            # Calculate distances to all class means in the transformed space
            distances: list[float] = [
                np.linalg.norm(sample - self.class_means_trans[c])
                for c in self.class_means_trans
            ]
            # Select the class with the minimum distance
            predicted_class: int = min(
                self.class_means_trans,
                key=lambda c: np.linalg.norm(sample - self.class_means_trans[c]),
            )
            predictions.append(predicted_class)

        return np.array(predictions)
