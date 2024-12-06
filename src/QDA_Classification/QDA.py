"""QDA Classification"""
import numpy as np

class QDA:
    """
    Quadratic Discriminant Analysis (QDA) classifier.

    Attributes:
        classes (np.ndarray): Unique class labels.
        means (dict): Mean vectors for each class.
        covariances (dict): Covariance matrices for each class.
        priors (dict): Prior probabilities for each class.
    """

    def __init__(self):
        """
        Initializes the QDA classifier.
        """
        self.classes: np.ndarray = None
        self.means: dict = {}
        self.covariances: dict = {}
        self.priors: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the QDA model to the data.

        Args:
            X (np.ndarray): Feature data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.covariances[c] = np.cov(X_c, rowvar=False)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the input data.

        Args:
            X (np.ndarray): Feature data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        predictions = []

        for sample in X:
            class_probs = []

            for c in self.classes:
                mean = self.means[c]
                cov = self.covariances[c]
                prior = self.priors[c]

                inv_cov = np.linalg.inv(cov)
                diff = sample - mean
                exponent = -0.5 * diff.T @ inv_cov @ diff
                norm_const = np.sqrt((2 * np.pi) ** len(mean) * np.linalg.det(cov))

                prob = np.exp(exponent) / norm_const
                class_probs.append(prior * prob)

            predictions.append(self.classes[np.argmax(class_probs)])

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class probabilities for the input data.

        Args:
            X (np.ndarray): Feature data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class probabilities of shape (n_samples, n_classes).
        """
        probas = []

        for sample in X:
            class_probs = []

            for c in self.classes:
                mean = self.means[c]
                cov = self.covariances[c]
                prior = self.priors[c]

                inv_cov = np.linalg.inv(cov)
                diff = sample - mean
                exponent = -0.5 * diff.T @ inv_cov @ diff
                norm_const = np.sqrt((2 * np.pi) ** len(mean) * np.linalg.det(cov))

                prob = np.exp(exponent) / norm_const
                class_probs.append(prior * prob)

            probas.append(class_probs)

        return np.array(probas)
