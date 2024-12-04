"""LDA with some datasets^^"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

from Step_by_Step import StepByStepExplain
from LDA import LDA

class LDATwoBlobs(StepByStepExplain):
    """
    LDATwoBlobs class for demonstrating LDA on a two-blobs dataset.
    Inherits from StepByStepExplain.

    Attributes:
        X (ndarray): Feature data.
        y (ndarray): Target labels.
    """

    def __init__(self):
        """
        Initializes the LDATwoBlobs class with a two-blobs dataset.
        """
        X,y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)
        super().__init__(X, y)
    
    def show_data_before_project_by_scatter(self):
        """
        Plots the original two-blobs dataset using a scatter plot.

        Args:
            None
        """
        X: ndarray = self.X
        y: ndarray = self.y
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.title('Two Moons Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def forward(self) -> tuple:
        """
        Performs LDA and returns the LDA direction and the transformed data for each class.

        Args:
            None

        Returns:
            tuple: A tuple containing the LDA direction, transformed data for class 0, and transformed data for class 1.
        """
        _, eigenvectors = self.step_by_step()
        X_LDA_Class0: ndarray = np.dot(self.X[self.y == 0], eigenvectors[:, 0])
        X_LDA_Class1: ndarray = np.dot(self.X[self.y == 1], eigenvectors[:, 0])

        lda_direction: ndarray = eigenvectors[:, 0]

        return lda_direction, X_LDA_Class0, X_LDA_Class1
    
    def plot_LDA_line(self) -> None:
        """
        Plots the LDA separation axis on the original two-blobs dataset.

        Args:
            None
        """
        X: ndarray = self.X
        y: ndarray = self.y
        lda_direction, _, _ = self.forward()
        plt.figure(figsize=(8, 6))
        u1: ndarray = np.mean(X[y == 0], axis=0)
        x_vals: ndarray = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        y_vals: ndarray = (lda_direction[1] / lda_direction[0]) * (x_vals - u1[0]) + u1[1]
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        # Plot LDA separation axis
        plt.plot(x_vals, y_vals, 'k--', label='LDA Axis')

        # Set title and show plot
        plt.title('Two Blobs Dataset with LDA Separation Axis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_data_after_projection(self) -> None:
        """
        Plots the data after LDA projection.

        Args:
            None
        """
        _, X_LDA_Class0, X_LDA_Class1 = self.forward()
        plt.figure(figsize=(8, 6))
        plt.scatter(X_LDA_Class0, np.zeros(len(X_LDA_Class0)), color='red', label='Class 0')
        plt.scatter(X_LDA_Class1, np.zeros(len(X_LDA_Class1)), color='blue', label='Class 1')
        plt.title("Data after Dimensionality Reduction")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        # Show plot
        plt.tight_layout()
        plt.show()

    def plot_distribution_data_after_projection(self) -> None:
        """
        Plots the distribution of the data after LDA projection.

        Args:
            None
        """
        _, w = self.step_by_step()
        
        X_projected: ndarray = np.dot(self.X, w)
        lda_data: pd.DataFrame = pd.DataFrame({
            'LDA Component': X_projected,  # LDA component after projection
            'Class': self.y  # Class labels
        })

        # Ensure 'Class' is categorical
        lda_data['Class'] = lda_data['Class'].astype('category')

        # Plot distribution with kdeplot
        plt.figure(figsize=(10, 6))

        # KDE Plot (distribution curve) - with 'clip' to limit the range of KDE
        sns.kdeplot(
            data=lda_data,
            x="LDA Component",  # LDA axis
            hue="Class",  # Color by class
            fill=True,  # Fill under the KDE curves
            common_norm=False,  # Separate classes to avoid common normalization
        )

        # Scatter Plot (data points)
        sns.scatterplot(
            data=lda_data,
            x="LDA Component",  # LDA axis
            y=np.zeros_like(lda_data["LDA Component"]),  # Points plotted at y = 0
            hue="Class",  # Color by class
            palette="Set2",  # Choose color palette
            marker="o",  # Choose point shape
            edgecolor='black',  # Edge color for points
            s=50,  # Point size
            legend=False  # Disable legend for data points
        )

        # Add title and show plot
        plt.title("Distribution of Data and Data Points after Dimensionality Reduction (LDA)", fontsize=16)
        plt.xlabel("LDA Component")
        plt.ylabel("Density")

        plt.show()

class IrisDataSetWithLDA(LDA):
    """
    LDAIrisDataSet class for demonstrating LDA on the Iris dataset.
    Inherits from StepByStepExplain.

    Attributes:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target labels.
        feature_names (list): Names of the features.
        target_names (list): Names of the target classes.
    """

    def __init__(self,k) -> None:
        """
        Initializes the LDAIrisDataSet class with the Iris dataset.
        """
        super().__init__(k)
        data = load_iris()
        self.X: np.ndarray = data.data
        self.y: np.ndarray = data.target
        self.feature_names: list = data.feature_names
        self.target_names: list = data.target_names

    def plot_first_two_feature(self) -> None:
        """
        Plots a scatter plot of the first two features of the Iris dataset.

        Args:
            None
        """
        plt.figure(figsize=(8, 6))
        for i, target_name in enumerate(self.target_names):
            plt.scatter(self.X[self.y == i, 0], self.X[self.y == i, 1], label=target_name)
        plt.title("Scatter plot of the first two features")
        plt.xlabel(self.feature_names[0])  # Sepal length
        plt.ylabel(self.feature_names[1])  # Sepal width
        plt.legend()
        plt.show()
        
    def visualize_data_with_pair_plot_bf_re_di(self) -> None:
        """
        Visualizes the Iris dataset with a pair plot before dimensionality reduction.

        Args:
            None
        """
        df: pd.DataFrame = pd.DataFrame(data=self.X, columns=self.feature_names)
        df['species'] = self.y
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        sns.pairplot(data=df, hue='species', diag_kind='kde')

    def visualize_data_with_pair_plot_and_kde_bf_re_di(self) ->None:
        """
        Visualizes the Iris dataset with a pair plot and KDE before dimensionality reduction.

        Args:
            None
        """
        df: pd.DataFrame = pd.DataFrame(data=self.X, columns=self.feature_names)
        df['species'] = self.y
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        sns.pairplot(data=df, hue='species').map_diag(sns.histplot).map_lower(sns.kdeplot).map_upper(sns.kdeplot)
    
    def visualize_data_with_joint_plot_bf_re_di(self) ->None:
        """
        Visualizes the Iris dataset with a joint plot before dimensionality reduction.

        Args:
            None
        """
        df: pd.DataFrame = pd.DataFrame(data=self.X, columns=self.feature_names)
        df['species'] = self.y
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        g = sns.jointplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
        g.plot_joint(sns.kdeplot)

    def forward(self) -> ndarray:
        """
        Fits the LDA model to the Iris dataset.

        Args:
            None
        """
        self.fit(self.X,self.y)
        X_projected = self.tranform(self.X)
        return X_projected

class LDAIrisDataSetOneComponentVisualize(IrisDataSetWithLDA):
    """
    LDAIrisDataSetOneComponentVisualize class for visualizing the Iris dataset with one LDA component.
    Inherits from IrisDataSetWithLDA.

    Attributes:
        X_projected (np.ndarray): Projected data after LDA transformation.
    """

    def __init__(self):
        """
        Initializes the LDAIrisDataSetOneComponentVisualize class with one LDA component.
        """
        super().__init__(1)
        self.X_projected: np.ndarray = self.forward()
    
    def visualize_x_projected(self):
        """
        Visualizes the projected data with one LDA component.

        Args:
            None
        """
        plt.scatter(self.X_projected, [0] * len(self.X_projected), c=self.y, edgecolor='none', alpha=0.8, cmap='viridis', vmin=0, vmax=2)
        plt.xlabel('Linear Discriminant 1')
        plt.colorbar(label="Class Labels")
        plt.yticks([])

        colors: list = [plt.cm.viridis(i / 3) for i in range(len(self.target_names))]

        for cls, color in zip(self.target_names, colors):
            plt.scatter([], [], c=[color], label=cls)

        plt.legend(title="Classes")
        plt.show()

class LDAIrisDataSetTwoComponentVisualize(IrisDataSetWithLDA):
    """
    LDAIrisDataSetTwoComponentVisualize class for visualizing the Iris dataset with two LDA components.
    Inherits from IrisDataSetWithLDA.

    Attributes:
        X_projected (np.ndarray): Projected data after LDA transformation.
    """

    def __init__(self):
        """
        Initializes the LDAIrisDataSetTwoComponentVisualize class with two LDA components.
        """
        super().__init__(2)
        self.X_projected: np.ndarray = self.forward()
    
    def visualize_x_projected(self):
        """
        Visualizes the projected data with two LDA components.

        Args:
            None
        """
        x1_tran: np.ndarray = self.X_projected[:, 0]
        x2_tran: np.ndarray = self.X_projected[:, 1]
        plt.colormaps['viridis']
        plt.scatter(x1_tran, x2_tran, c=self.y, edgecolor='none', alpha=0.8, cmap='viridis', vmin=0, vmax=2)

        plt.xlabel('Linear Discriminant 1')
        plt.ylabel('Linear Discriminant 2')
        plt.colorbar(label="Class Labels")
        plt.show()
