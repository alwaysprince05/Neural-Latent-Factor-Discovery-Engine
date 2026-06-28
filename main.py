"""
main.py
Entry point for Neural Latent Factor Discovery Engine
"""
import argparse
import numpy as np
import pandas as pd
from data import get_data
from pca_model import PCAModel
from autoencoder import Autoencoder
from training import train_autoencoder
import visualization as viz
import torch


def main():
    parser = argparse.ArgumentParser(description='Neural Latent Factor Discovery Engine')
    parser.add_argument('--source', choices=['csv', 'yfinance'], required=True, help='Data source')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--tickers', nargs='+', help='List of tickers for yfinance')
    parser.add_argument('--start', type=str, help='Start date for yfinance')
    parser.add_argument('--end', type=str, help='End date for yfinance')
    parser.add_argument('--pca_k', type=int, default=3, help='Number of PCA components')
    parser.add_argument('--ae_hidden', nargs='+', type=int, default=[32,16], help='Autoencoder hidden layer sizes')
    parser.add_argument('--ae_latent', type=int, default=3, help='Autoencoder latent dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Autoencoder training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Autoencoder batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device for training (cpu or cuda)')
    args = parser.parse_args()

    # Data pipeline
    if args.source == 'csv':
        log_ret, std_ret = get_data('csv', args.csv_path)
    else:
        log_ret, std_ret = get_data('yfinance', args.tickers, args.start, args.end)
    asset_names = std_ret.columns.tolist()

    # Classical PCA
    pca = PCAModel(n_components=args.pca_k)
    pca.fit(std_ret)
    pca_latents = pca.transform(std_ret)
    explained_var = pca.explained_variance_ratio()
    loadings = pca.loading_matrix()

    # Deep Autoencoder
    model, train_losses, val_losses, ae_latents = train_autoencoder(
        std_ret, input_dim=std_ret.shape[1], hidden_dims=args.ae_hidden,
        latent_dim=args.ae_latent, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    # Analysis
    # 1. Compare reconstruction error
    pca_recon = np.dot(pca_latents, loadings.T)
    pca_recon_error = np.mean((std_ret.values - pca_recon)**2)
    ae_recon_error = val_losses[-1]
    print(f"PCA Reconstruction Error: {pca_recon_error:.6f}")
    print(f"Autoencoder Reconstruction Error: {ae_recon_error:.6f}")

    # 2. Explained variance
    print("PCA Explained Variance Ratio:", explained_var)
    # For autoencoder, use variance of latent factors as proxy
    ae_latent_var = np.var(ae_latents, axis=0)
    print("Autoencoder Latent Variance:", ae_latent_var)

    # 3. Stability over time (plot latent factor evolution)

    # Visualization
    # 1. 3D latent space projection
    if args.ae_latent >= 3:
        viz.plot_3d_latent(ae_latents, title='Autoencoder 3D Latent Space')
    if args.pca_k >= 3:
        viz.plot_3d_latent(pca_latents, title='PCA 3D Latent Space')
    # 2. Factor loading surface plot
    viz.plot_loading_surface(loadings, asset_names, title='PCA Loading Surface')
    # 3. Reconstruction error curve
    viz.plot_reconstruction_error(train_losses, val_losses)
    # 4. Explained variance bar chart
    viz.plot_explained_variance_bar(explained_var)
    # 5. Latent factor time evolution plot
    viz.plot_latent_evolution(ae_latents, title='Autoencoder Latent Evolution')
    viz.plot_latent_evolution(pca_latents, title='PCA Latent Evolution')

if __name__ == '__main__':
    main()
