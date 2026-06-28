"""
visualization.py
Visualization utilities for latent factor analysis.
Dark scientific theme, high-res figures.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

plt.style.use('dark_background')
sns.set_theme(style='darkgrid')

# 1. 3D latent space projection
def plot_3d_latent(latents, title='3D Latent Space Projection'):
    fig = plt.figure(figsize=(10,8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latents[:,0], latents[:,1], latents[:,2], c='cyan', s=40, alpha=0.7)
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_zlabel('Latent 3')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# 2. Factor loading surface plot
def plot_loading_surface(loadings, asset_names=None, title='Factor Loading Surface'):
    fig = plt.figure(figsize=(12,7), dpi=150)
    ax = fig.add_subplot(111)
    cax = ax.imshow(loadings, aspect='auto', cmap='plasma')
    fig.colorbar(cax, ax=ax, label='Loading Value')
    ax.set_xlabel('Component')
    ax.set_ylabel('Asset')
    if asset_names is not None:
        ax.set_yticks(np.arange(len(asset_names)))
        ax.set_yticklabels(asset_names)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# 3. Reconstruction error curve
def plot_reconstruction_error(train_losses, val_losses, title='Reconstruction Error Curve'):
    fig = plt.figure(figsize=(10,6), dpi=150)
    plt.plot(train_losses, label='Train Loss', color='lime')
    plt.plot(val_losses, label='Validation Loss', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. Explained variance bar chart
def plot_explained_variance_bar(explained_var, title='Explained Variance Ratio'):
    fig = plt.figure(figsize=(10,6), dpi=150)
    plt.bar(np.arange(len(explained_var)), explained_var, color='deepskyblue')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 5. Latent factor time evolution plot
def plot_latent_evolution(latents, title='Latent Factor Time Evolution'):
    fig = plt.figure(figsize=(12,7), dpi=150)
    for i in range(latents.shape[1]):
        plt.plot(latents[:,i], label=f'Latent {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Latent Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
