"""
pca_model.py
Classical PCA model for latent factor extraction.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAModel:
    def __init__(self, n_components: int):
        """Initialize PCA model."""
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        """Fit PCA to standardized returns."""
        self.pca.fit(X)
        self.fitted = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Project data onto principal components."""
        return self.pca.transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        return self.pca.explained_variance_ratio_

    def loading_matrix(self) -> np.ndarray:
        """Return PCA loading matrix (components)."""
        return self.pca.components_.T

    def plot_loadings(self, asset_names=None, figsize=(10,6), cmap='viridis'):
        """Visualize loading matrix as a surface plot."""
        if not self.fitted:
            raise RuntimeError("PCA model not fitted.")
        loadings = self.loading_matrix()
        plt.figure(figsize=figsize)
        plt.imshow(loadings, aspect='auto', cmap=cmap)
        plt.colorbar(label='Loading Value')
        plt.xlabel('Component')
        plt.ylabel('Asset')
        if asset_names is not None:
            plt.yticks(np.arange(len(asset_names)), asset_names)
        plt.title('PCA Loading Matrix')
        plt.tight_layout()
        plt.show()
