import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from data import get_data
from pca_model import PCAModel
from training import train_autoencoder

# Set page configuration with wide layout
st.set_page_config(
    page_title="Neural Latent Factor Discovery Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for state-of-the-art dark glassmorphic design and animations
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #15102a 0%, #07060f 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0b0a14;
        border-right: 1px solid #1c1a30;
    }
    
    /* Custom Headers */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        font-size: 3.2rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 35%, #b180ff 70%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        font-weight: 800;
        text-align: center;
        filter: drop-shadow(0 0 30px rgba(79, 172, 254, 0.2));
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
        text-align: center;
    }
    
    /* Metric Cards with Glassmorphism & Neon Glow */
    .metric-container {
        display: flex;
        gap: 24px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(17, 16, 32, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
    }
    
    .pca-card::before {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }
    
    .ae-card::before {
        background: linear-gradient(90deg, #b180ff 0%, #f472b6 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .pca-card:hover {
        box-shadow: 0 15px 45px rgba(0, 242, 254, 0.15);
    }
    
    .ae-card:hover {
        box-shadow: 0 15px 45px rgba(177, 128, 255, 0.15);
    }
    
    .metric-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-top: 12px;
    }
    
    .pca-val {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .ae-val {
        background: linear-gradient(135deg, #f472b6 0%, #b180ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-lbl {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
    }
    
    /* Clean Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
        margin: 40px 0;
    }
    
    /* Custom tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(177, 128, 255, 0.15) 100%) !important;
        border-color: rgba(79, 172, 254, 0.4) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">Neural Latent Factor Discovery Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extracting deep hidden factors from financial asset returns with high-dimensional modeling</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### 🛠️ Configuration")

# 1. Data Source Selection
source = st.sidebar.selectbox("Data Source", ["Yahoo Finance (Demo/API)", "Upload CSV"])

if source == "Yahoo Finance (Demo/API)":
    tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, GOOG, AMZN, TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    col_d1, col_d2 = st.sidebar.columns(2)
    start_date = col_d1.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = col_d2.date_input("End Date", pd.to_datetime("2024-01-01"))
else:
    uploaded_file = st.sidebar.file_uploader("Upload Price CSV (First column as Date)", type=["csv"])
    tickers = None
    start_date, end_date = None, None

# 2. PCA Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 📉 PCA Settings")
pca_k = st.sidebar.slider("Number of PCA Components (K)", min_value=2, max_value=10, value=3)

# 3. Autoencoder Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Autoencoder Settings")
ae_latent = st.sidebar.slider("Latent Dimension", min_value=2, max_value=10, value=3)
hidden_dims_input = st.sidebar.text_input("Hidden Layer Sizes", "32, 16")
try:
    ae_hidden = [int(h.strip()) for h in hidden_dims_input.split(",")]
except ValueError:
    st.sidebar.error("Invalid hidden layer input. Resetting to 32, 16")
    ae_hidden = [32, 16]

col_a1, col_a2 = st.sidebar.columns(2)
epochs = col_a1.number_input("Epochs", min_value=10, max_value=1000, value=100)
batch_size = col_a2.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lr = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

st.sidebar.markdown(" ")
run_button = st.sidebar.button("🚀 Run Analysis Engine", use_container_width=True)

# Main Execution Flow
if run_button or 'run_done' in st.session_state:
    st.session_state['run_done'] = True
    
    with st.spinner("Executing discovery engine and training models..."):
        # Load and compute returns
        try:
            if source == "Yahoo Finance (Demo/API)":
                log_ret, std_ret = get_data('yfinance', tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            else:
                if uploaded_file is None:
                    st.error("Please upload a CSV file first.")
                    st.stop()
                df_uploaded = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                log_ret = np.log(df_uploaded / df_uploaded.shift(1)).dropna()
                st.session_state['asset_names'] = log_ret.columns.tolist()
                std_ret = (log_ret - log_ret.mean()) / log_ret.std()
                
            asset_names = std_ret.columns.tolist()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Run Models
        # 1. Classical PCA
        pca = PCAModel(n_components=pca_k)
        pca.fit(std_ret)
        pca_latents = pca.transform(std_ret)
        explained_var = pca.explained_variance_ratio()
        loadings = pca.loading_matrix()

        # 2. Deep Autoencoder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, train_losses, val_losses, ae_latents = train_autoencoder(
            std_ret, input_dim=std_ret.shape[1], hidden_dims=ae_hidden,
            latent_dim=ae_latent, epochs=epochs, batch_size=batch_size, lr=lr, device=device
        )

        # 3. Compute Reconstruction Errors
        pca_recon = np.dot(pca_latents, loadings.T)
        pca_recon_error = np.mean((std_ret.values - pca_recon)**2)
        ae_recon_error = val_losses[-1]

    # Metrics Section with stunning cards
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card pca-card">
            <div class="metric-lbl">Classical PCA Reconstruction Error (MSE)</div>
            <div class="metric-val pca-val">{pca_recon_error:.6f}</div>
        </div>
        <div class="metric-card ae-card">
            <div class="metric-lbl">Deep Autoencoder Reconstruction Error (MSE)</div>
            <div class="metric-val ae-val">{ae_recon_error:.6f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Tabs for visualization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 3D Latent Projections", 
        "📊 Factor Loadings Matrix", 
        "📈 Latent Factor Evolution", 
        "🎯 Model Performance & Variance", 
        "📋 Data Preview"
    ])

    # Plotly Custom Premium Dark Style Config
    plotly_layout_args = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Plus Jakarta Sans'),
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(gridcolor='#1b1932', zerolinecolor='#2a274c'),
        yaxis=dict(gridcolor='#1b1932', zerolinecolor='#2a274c')
    )

    with tab1:
        st.subheader("3D Latent Space Projection")
        st.markdown("Spin and zoom to explore how days map in the 3D latent space.")
        t_col1, t_col2 = st.columns(2)
        
        # PCA 3D Latent Plot
        if pca_k >= 3:
            fig_pca_3d = px.scatter_3d(
                x=pca_latents[:, 0], y=pca_latents[:, 1], z=pca_latents[:, 2],
                color=np.arange(len(pca_latents)),
                labels={'x': 'Comp 1', 'y': 'Comp 2', 'z': 'Comp 3', 'color': 'Time Index'},
                title="PCA Latent Projections",
                color_continuous_scale="Tealrose"
            )
            fig_pca_3d.update_layout(plotly_layout_args)
            t_col1.plotly_chart(fig_pca_3d, use_container_width=True)
        else:
            t_col1.warning("Set PCA Components (K) >= 3 to view 3D projection.")

        # Autoencoder 3D Latent Plot
        if ae_latent >= 3:
            fig_ae_3d = px.scatter_3d(
                x=ae_latents[:, 0], y=ae_latents[:, 1], z=ae_latents[:, 2],
                color=np.arange(len(ae_latents)),
                labels={'x': 'Latent 1', 'y': 'Latent 2', 'z': 'Latent 3', 'color': 'Time Index'},
                title="Autoencoder Latent Projections",
                color_continuous_scale="Viridis"
            )
            fig_ae_3d.update_layout(plotly_layout_args)
            t_col2.plotly_chart(fig_ae_3d, use_container_width=True)
        else:
            t_col2.warning("Set Autoencoder Latent Dimension >= 3 to view 3D projection.")

    with tab2:
        st.subheader("PCA Factor Loadings Matrix")
        st.markdown("Check how each asset contributes to the principal components.")
        
        fig_loadings = go.Figure(data=go.Heatmap(
            z=loadings,
            x=[f"Comp {i+1}" for i in range(loadings.shape[1])],
            y=asset_names,
            colorscale='Turbo',
            colorbar=dict(title='Loading')
        ))
        fig_loadings.update_layout(
            plotly_layout_args,
            title="PCA Loading Coefficients Matrix",
            height=600
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

    with tab3:
        st.subheader("Latent Factor Time Series")
        st.markdown("Analyze how the extracted risk factors fluctuate over time.")
        
        # PCA Evolution
        df_pca_latents = pd.DataFrame(pca_latents, index=std_ret.index, columns=[f"Component {i+1}" for i in range(pca_k)])
        fig_pca_ev = px.line(
            df_pca_latents, 
            title="PCA Latent Factors Evolution",
            labels={'value': 'Value', 'index': 'Date'},
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_pca_ev.update_traces(line_shape='spline', line_width=2)
        fig_pca_ev.update_layout(plotly_layout_args)
        st.plotly_chart(fig_pca_ev, use_container_width=True)

        # Autoencoder Evolution
        df_ae_latents = pd.DataFrame(ae_latents, index=std_ret.index, columns=[f"Latent {i+1}" for i in range(ae_latent)])
        fig_ae_ev = px.line(
            df_ae_latents, 
            title="Autoencoder Latent Factors Evolution",
            labels={'value': 'Value', 'index': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_ae_ev.update_traces(line_shape='spline', line_width=2)
        fig_ae_ev.update_layout(plotly_layout_args)
        st.plotly_chart(fig_ae_ev, use_container_width=True)

    with tab4:
        st.subheader("Model Performance & Variance")
        p_col1, p_col2 = st.columns(2)
        
        # Autoencoder Loss Curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss', line=dict(color='#00ffcc', width=2.5, shape='spline')))
        fig_loss.add_trace(go.Scatter(y=val_losses, mode='lines', name='Val Loss', line=dict(color='#ff00ff', width=2.5, shape='spline')))
        fig_loss.update_layout(
            plotly_layout_args,
            title="Autoencoder Training Loss (MSE)",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        p_col1.plotly_chart(fig_loss, use_container_width=True)

        # PCA Variance Explained
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"Comp {i+1}" for i in range(len(explained_var))],
            y=explained_var,
            marker=dict(color='#4facfe', line=dict(color='#00f2fe', width=1.5)),
            name='Explained Variance'
        ))
        fig_var.update_layout(
            plotly_layout_args,
            title="PCA Explained Variance Ratio per Component",
            xaxis_title="Component",
            yaxis_title="Ratio"
        )
        p_col2.plotly_chart(fig_var, use_container_width=True)

    with tab5:
        st.subheader("Standardized Log Returns Data")
        st.dataframe(std_ret, use_container_width=True)

else:
    # Beautiful welcome screen / placeholder
    st.markdown("""
    <div style="background: rgba(17, 16, 32, 0.4); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 24px; padding: 60px; text-align: center; margin-top: 50px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);">
        <span style="font-size: 5.5rem; filter: drop-shadow(0 0 20px rgba(177, 128, 255, 0.5));">🔮</span>
        <h2 style="color: #ffffff; margin-top: 25px; font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem;">Ready to Discover Hidden Latent Factors?</h2>
        <p style="color: #94a3b8; max-width: 650px; margin: 15px auto 35px auto; font-size: 1.1rem; line-height: 1.7; font-weight: 300;">
            Configure your tickers, data range, and model details in the left sidebar, then click <b>Run Analysis Engine</b> to build the Classical PCA and Deep PyTorch Autoencoder models.
        </p>
    </div>
    """, unsafe_allow_html=True)
