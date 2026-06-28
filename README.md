
# Neural Latent Factor Discovery Engine

**Creator/Dev:** Prince Maurya

## What is this project about?
This project is a scientific Python engine for discovering hidden (latent) risk factors in financial asset returns. It uses both classical Principal Component Analysis (PCA) and modern deep learning autoencoders to extract, analyze, and visualize latent factors that drive asset price movements. The system is designed for quantitative finance research, risk management, and machine learning experimentation.

## File Overview
- `data.py`: Data pipeline for loading historical prices (CSV or Yahoo Finance), computing log returns, and standardizing data.
- `pca_model.py`: Implements classical PCA for extracting top K components, explained variance, and loading matrix visualization.
- `autoencoder.py`: PyTorch deep autoencoder model for nonlinear latent factor extraction.
- `training.py`: Utilities for training the autoencoder, including train/validation split and loss tracking.
- `visualization.py`: High-resolution, dark-themed scientific plots for latent space, loadings, error curves, and more.
- `main.py`: Main entry point to run the full pipeline, compare models, and generate all analyses and visualizations.

## How to Fork and Use
1. **Fork this repository** on GitHub to your own account.
2. **Clone your fork** to your local machine:
	```bash
	git clone https://github.com/alwaysprince05/Neural-Latent-Factor-Discovery-Engine.git
	cd Neural-Latent-Factor-Discovery-Engine
	```
3. **Install dependencies** (in a virtual environment recommended):
	```bash
	pip install numpy pandas yfinance scikit-learn torch matplotlib seaborn
	```
4. **Run the engine** with your own data or Yahoo Finance tickers:
	```bash
	python main.py --source csv --csv_path data/your_prices.csv --pca_k 3 --ae_hidden 32 16 --ae_latent 3 --epochs 100
	# Or with Yahoo Finance
	python main.py --source yfinance --tickers AAPL MSFT GOOG --start 2020-01-01 --end 2023-01-01 --pca_k 3 --ae_hidden 32 16 --ae_latent 3 --epochs 100
	```

## Features
- Data pipeline: CSV or Yahoo Finance, log returns, standardization
- Classical PCA: top K components, explained variance, loading visualization
- Deep Autoencoder: PyTorch, configurable layers, latent extraction
- Analysis: compare PCA vs autoencoder (reconstruction error, explained variance, stability)
- Visualization: 3D latent space, loading surface, error curve, variance bar, latent evolution
- Dark scientific theme, high-res figures

## Notes
- All figures use a dark scientific theme and are high-resolution.
- Modular, well-commented code for research and extension.
