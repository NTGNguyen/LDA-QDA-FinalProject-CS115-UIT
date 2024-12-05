"""LDA Classification with two types of datasets"""
from ..LDA import LDA

from sklearn.utils._bunch import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from pandas import Series
from typing import Tuple
from numpy.typing import NDArray
from numpy import Float64
from typing import Any
from sklearn.preprocessing import LabelEncoder

class LDAClassificationsSeaborns(LDA):
    """
    LDAClassificationsSeaborns class for demonstrating LDA on a Seaborn dataset.
    Inherits from LDA.

    Attributes:
        dataset (DataFrame): The dataset to be used.
        X (NDArray[Float64]): Feature data.
        y (NDArray[Any]): Target labels.
    """

    def __init__(self, k: int, dataset: DataFrame):
        """
        Initializes the LDAClassificationsSeaborns class with the specified number of components and dataset.

        Args:
            k (int): Number of components to keep.
            dataset (DataFrame): The dataset to be used.
        """
        super().__init__(k)
        self.dataset: DataFrame = dataset
        self.X: NDArray[Float64] = None
        self.y: NDArray[Any] = None

    def preprocess(self):
        """
        Preprocesses the dataset by encoding the labels and splitting the data into features and target.
        """
        # Assuming the last column is the target
        X = self.dataset.iloc[:, :-1].values
        y = self.dataset.iloc[:, -1].values

        # Encode the target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        self.X = X
        self.y = y_encoded

    def forward_and_predict(self) -> float:
        """
        Splits the data, fits the LDA model, and makes predictions.

        Returns:
            float: The accuracy of the model.
        """
        X_train: NDArray[Float64]
        X_test: NDArray[Float64]
        y_train: NDArray[Any]
        y_test: NDArray[Any]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.fit(X_train, y_train)
        y_pred: NDArray[Any] = self.predict(X_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        return accuracy

class LDAClassificationsSKLearns(LDA):
    """
    LDAClassificationsSKLearns class for demonstrating LDA on a dataset from sklearn.
    Inherits from LDA.

    Attributes:
        dataset (Tuple[NDArray[Float64], NDArray[Any]]): The dataset to be used.
    """

    def __init__(self, k: int, dataset: Tuple[NDArray[Float64], NDArray[Any]]):
        """
        Initializes the LDAClassificationsSKLearns class with the specified number of components and dataset.

        Args:
            k (int): Number of components to keep.
            dataset (Tuple[NDArray[Float64], NDArray[Any]]): The dataset to be used.
        """
        super().__init__(k)
        self.dataset: Tuple[NDArray[Float64], NDArray[Any]] = dataset
    
    def forward_and_predict(self) -> float:
        """
        Splits the data, fits the LDA model, and makes predictions.

        Returns:
            float: The accuracy of the model.
        """
        X_train: NDArray[Float64]
        X_test: NDArray[Float64]
        y_train: NDArray[Any]
        y_test: NDArray[Any]

        X_train, X_test, y_train, y_test = train_test_split(self.dataset[0], self.dataset[1], test_size=0.3, random_state=42)
        self.fit(X_train, y_train)
        y_pred: NDArray[Any] = self.predict(X_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        return accuracy

class LDAClassificationsSKLearns(LDA):
    """
    LDAClassificationsSKLearns class for demonstrating LDA on a dataset from sklearn.
    Inherits from LDA.

    Attributes:
        dataset (Tuple[NDArray[Float64], NDArray[Any]]): The dataset to be used.
    """

    def __init__(self, k: int, dataset: Tuple[NDArray[Float64], NDArray[Any]]):
        """
        Initializes the LDAClassificationsSKLearns class with the specified number of components and dataset.

        Args:
            k (int): Number of components to keep.
            dataset (Tuple[NDArray[Float64], NDArray[Any]]): The dataset to be used.
        """
        super().__init__(k)
        self.dataset: Tuple[NDArray[Float64], NDArray[Any]] = dataset
    
    def forward_and_predict(self) -> float:
        """
        Splits the data, fits the LDA model, and makes predictions.

        Returns:
            float: The accuracy of the model.
        """
        X_train: NDArray[Float64]
        X_test: NDArray[Float64]
        y_train: NDArray[Any]
        y_test: NDArray[Any]

        X_train, X_test, y_train, y_test = train_test_split(self.dataset[0], self.dataset[1], test_size=0.3, random_state=42)
        self.fit(X_train, y_train)
        y_pred: NDArray[Any] = self.predict(X_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        return accuracy



