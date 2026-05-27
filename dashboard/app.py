"""
Streamlit dashboard for visualizing LLM benchmark results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="NPU LLM Benchmark Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chart-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load benchmark results from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "benchmark_results.json"
    
    if data_path.exists():
        with open(data_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

def create_comparison_chart(df, precision, metric, title_suffix):
    """Create a bar chart comparing CPU, GPU, and NPU for a specific metric."""
    # Filter data for the precision
    df_filtered = df[df['precision'] == precision]
    
    if df_filtered.empty:
        return None
    
    # Map metric names to dataframe columns
    metric_mapping = {
        'latency': 'latency_ms_mean',
        'token_generation': 'tokens_per_second_mean',
        'throughput': 'throughput_mean'
    }
    
    metric_col = metric_mapping.get(metric, metric)
    
    # Group by hardware
    df_grouped = df_filtered.groupby('hardware')[metric_col].mean().reset_index()
    
    # Define colors for hardware
    color_map = {
        'cpu': '#3498db',
        'gpu': '#e74c3c',
        'npu': '#2ecc71'
    }
    
    colors = [color_map.get(hw, '#95a5a6') for hw in df_grouped['hardware']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_grouped['hardware'],
            y=df_grouped[metric_col],
            marker_color=colors,
            text=df_grouped[metric_col].round(2),
            textposition='auto',
        )
    ])
    
    # Update layout
    metric_labels = {
        'latency': 'Latency (ms)',
        'token_generation': 'Token Generation (tokens/sec)',
        'throughput': 'Throughput (tokens/sec)'
    }
    
    fig.update_layout(
        title=f"{metric_labels[metric]} - {precision.upper()} {title_suffix}",
        xaxis_title="Hardware",
        yaxis_title=metric_labels[metric],
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def main():
    """Main dashboard application."""
    st.title("🚀 LLM Benchmark Dashboard")
    
    # Model information dictionary
    model_info = {
        'gpt2-fp32': {'name': 'GPT-2 (Small)', 'params': '117M', 'size': '548MB', 'precision': 'FP32'},
        'gpt2-fp16': {'name': 'GPT-2 (Small)', 'params': '117M', 'size': '548MB', 'precision': 'FP16'},
        'gpt2-int8': {'name': 'GPT-2 (Small)', 'params': '117M', 'size': '548MB', 'precision': 'INT8'},
        'gpt2-medium-fp32': {'name': 'GPT-2 (Medium)', 'params': '345M', 'size': '1.4GB', 'precision': 'FP32'},
        'gpt2-medium-fp16': {'name': 'GPT-2 (Medium)', 'params': '345M', 'size': '1.4GB', 'precision': 'FP16'},
        'gpt2-medium-int8': {'name': 'GPT-2 (Medium)', 'params': '345M', 'size': '1.4GB', 'precision': 'INT8'},
        'pythia-410m-fp32': {'name': 'Pythia 410M', 'params': '410M', 'size': '1.6GB', 'precision': 'FP32'},
        'pythia-410m-fp16': {'name': 'Pythia 410M', 'params': '410M', 'size': '1.6GB', 'precision': 'FP16'},
        'pythia-410m-int8': {'name': 'Pythia 410M', 'params': '410M', 'size': '1.6GB', 'precision': 'INT8'},
        'pythia-800m-fp32': {'name': 'Pythia 800M', 'params': '800M', 'size': '3.1GB', 'precision': 'FP32'},
        'pythia-800m-fp16': {'name': 'Pythia 800M', 'params': '800M', 'size': '3.1GB', 'precision': 'FP16'},
        'pythia-800m-int8': {'name': 'Pythia 800M', 'params': '800M', 'size': '3.1GB', 'precision': 'INT8'},
        'tinyllama-1.1b-fp32': {'name': 'TinyLlama 1.1B', 'params': '1.1B', 'size': '4.3GB', 'precision': 'FP32'},
        'tinyllama-1.1b-fp16': {'name': 'TinyLlama 1.1B', 'params': '1.1B', 'size': '4.3GB', 'precision': 'FP16'},
        'tinyllama-1.1b-int8': {'name': 'TinyLlama 1.1B', 'params': '1.1B', 'size': '4.3GB', 'precision': 'INT8'},
    }
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No benchmark data found. Please run the benchmark script first.")
        st.info("Run `python scripts/run_benchmarks.py` on your remote desktop to generate benchmark data.")
        return
    
    # Create tabs for model selection
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["GPT-2 Small", "GPT-2 Medium", "Pythia 410M", "Pythia 800M", "TinyLlama 1.1B"])
    
    # GPT-2 Small tab
    with tab1:
        # Model Details Section
        st.header("Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", "GPT-2 (Small)")
        
        with col2:
            st.metric("Parameters", "117M")
        
        with col3:
            st.metric("Model Size", "548MB")
        
        st.info("**Precision Variants:** FP32, FP16, INT8")
        
        st.markdown("---")
        
        # Filter data for GPT-2 Small models
        df_small = df[df['model'].str.contains('gpt2-', case=False, na=False) & ~df['model'].str.contains('medium', case=False, na=False)]
        
        if not df_small.empty:
            # Hardware Comparisons by Precision
            st.header("Hardware Comparisons by Precision")
            
            # Create tabs for each precision
            tab_small_int8, tab_small_fp16, tab_small_fp32 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
            
            with tab_small_int8:
                st.subheader("INT8 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_small, 'int8', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_small, 'int8', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_small, 'int8', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_small_fp16:
                st.subheader("FP16 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_small, 'fp16', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_small, 'fp16', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_small, 'fp16', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_small_fp32:
                st.subheader("FP32 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_small, 'fp32', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_small, 'fp32', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_small, 'fp32', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
        else:
            st.warning("No benchmark data found for GPT-2 Small models.")
    
    # GPT-2 Medium tab
    with tab2:
        # Model Details Section
        st.header("Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", "GPT-2 (Medium)")
        
        with col2:
            st.metric("Parameters", "345M")
        
        with col3:
            st.metric("Model Size", "1.4GB")
        
        st.info("**Precision Variants:** FP32, FP16, INT8")
        
        st.markdown("---")
        
        # Filter data for GPT-2 Medium models
        df_medium = df[df['model'].str.contains('medium', case=False, na=False)]
        
        if not df_medium.empty:
            # Hardware Comparisons by Precision
            st.header("Hardware Comparisons by Precision")
            
            # Create tabs for each precision
            tab_medium_int8, tab_medium_fp16, tab_medium_fp32 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
            
            with tab_medium_int8:
                st.subheader("INT8 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_medium, 'int8', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_medium, 'int8', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_medium, 'int8', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_medium_fp16:
                st.subheader("FP16 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_medium, 'fp16', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_medium, 'fp16', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_medium, 'fp16', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_medium_fp32:
                st.subheader("FP32 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_medium, 'fp32', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_medium, 'fp32', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_medium, 'fp32', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
        else:
            st.warning("No benchmark data found for GPT-2 Medium models.")
            st.info("Run `python scripts/download_model.py` and `python scripts/run_benchmarks.py` to generate GPT-2 Medium benchmarks.")
    
    # Pythia 410M tab
    with tab3:
        # Model Details Section
        st.header("Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", "Pythia 410M")
        
        with col2:
            st.metric("Parameters", "410M")
        
        with col3:
            st.metric("Model Size", "1.6GB")
        
        st.info("**Precision Variants:** FP32, FP16, INT8")
        
        st.markdown("---")
        
        # Filter data for Pythia 410M models
        df_pythia = df[df['model'].str.contains('pythia', case=False, na=False)]
        
        if not df_pythia.empty:
            # Hardware Comparisons by Precision
            st.header("Hardware Comparisons by Precision")
            
            # Create tabs for each precision
            tab_pythia_int8, tab_pythia_fp16, tab_pythia_fp32 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
            
            with tab_pythia_int8:
                st.subheader("INT8 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia, 'int8', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia, 'int8', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia, 'int8', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_pythia_fp16:
                st.subheader("FP16 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia, 'fp16', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia, 'fp16', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia, 'fp16', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_pythia_fp32:
                st.subheader("FP32 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia, 'fp32', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia, 'fp32', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia, 'fp32', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
        else:
            st.warning("No benchmark data found for Pythia 410M models.")
            st.info("Run `python scripts/download_model.py` and `python scripts/run_benchmarks.py` to generate Pythia 410M benchmarks.")
    
    # Pythia 800M tab
    with tab4:
        # Model Details Section
        st.header("Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", "Pythia 800M")
        
        with col2:
            st.metric("Parameters", "800M")
        
        with col3:
            st.metric("Model Size", "3.1GB")
        
        st.info("**Precision Variants:** FP32, FP16, INT8")
        
        st.markdown("---")
        
        # Filter data for Pythia 800M models
        df_pythia_800 = df[df['model'].str.contains('pythia-800m', case=False, na=False)]
        
        if not df_pythia_800.empty:
            # Hardware Comparisons by Precision
            st.header("Hardware Comparisons by Precision")
            
            # Create tabs for each precision
            tab_pythia800_int8, tab_pythia800_fp16, tab_pythia800_fp32 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
            
            with tab_pythia800_int8:
                st.subheader("INT8 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia_800, 'int8', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia_800, 'int8', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia_800, 'int8', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_pythia800_fp16:
                st.subheader("FP16 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia_800, 'fp16', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia_800, 'fp16', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia_800, 'fp16', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_pythia800_fp32:
                st.subheader("FP32 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_pythia_800, 'fp32', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_pythia_800, 'fp32', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_pythia_800, 'fp32', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
        else:
            st.warning("No benchmark data found for Pythia 800M models.")
            st.info("Run `python scripts/download_model.py` and `python scripts/run_benchmarks.py` to generate Pythia 800M benchmarks.")
    
    # TinyLlama 1.1B tab
    with tab5:
        # Model Details Section
        st.header("Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name", "TinyLlama 1.1B")
        
        with col2:
            st.metric("Parameters", "1.1B")
        
        with col3:
            st.metric("Model Size", "4.3GB")
        
        st.info("**Precision Variants:** FP32, FP16, INT8")
        
        st.markdown("---")
        
        # Filter data for TinyLlama 1.1B models
        df_tinyllama = df[df['model'].str.contains('tinyllama', case=False, na=False)]
        
        if not df_tinyllama.empty:
            # Hardware Comparisons by Precision
            st.header("Hardware Comparisons by Precision")
            
            # Create tabs for each precision
            tab_tinyllama_int8, tab_tinyllama_fp16, tab_tinyllama_fp32 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
            
            with tab_tinyllama_int8:
                st.subheader("INT8 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_tinyllama, 'int8', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_tinyllama, 'int8', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_tinyllama, 'int8', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_tinyllama_fp16:
                st.subheader("FP16 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_tinyllama, 'fp16', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_tinyllama, 'fp16', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_tinyllama, 'fp16', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
            
            with tab_tinyllama_fp32:
                st.subheader("FP32 Precision Comparisons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_latency = create_comparison_chart(df_tinyllama, 'fp32', 'latency', '')
                    if fig_latency:
                        st.plotly_chart(fig_latency, use_container_width=True)
                
                with col2:
                    fig_token = create_comparison_chart(df_tinyllama, 'fp32', 'token_generation', '')
                    if fig_token:
                        st.plotly_chart(fig_token, use_container_width=True)
                
                with col3:
                    fig_throughput = create_comparison_chart(df_tinyllama, 'fp32', 'throughput', '')
                    if fig_throughput:
                        st.plotly_chart(fig_throughput, use_container_width=True)
        else:
            st.warning("No benchmark data found for TinyLlama 1.1B models.")
            st.info("Run `python scripts/download_model.py` and `python scripts/run_benchmarks.py` to generate TinyLlama 1.1B benchmarks.")
    
    # Display summary statistics
    st.header("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_runs = len(df)
        st.metric("Total Benchmark Runs", total_runs)
    
    with col2:
        unique_configs = df[['precision', 'hardware']].drop_duplicates().shape[0]
        st.metric("Unique Configurations", unique_configs)
    
    with col3:
        models_tested = df['model'].nunique()
        st.metric("Models Tested", models_tested)
    
    st.markdown("---")
    
    # Display the 9 comparison charts as requested
    st.header("📈 Hardware Comparisons by Precision")
    
    precisions = ['int8', 'fp16', 'fp32']
    metrics = ['latency', 'token_generation', 'throughput']
    
    # Create tabs for each precision
    tab1, tab2, tab3 = st.tabs(["INT8 Precision", "FP16 Precision", "FP32 Precision"])
    
    with tab1:
        st.subheader("INT8 Precision Comparisons")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_latency = create_comparison_chart(df, 'int8', 'latency', '')
            if fig_latency:
                st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            fig_token = create_comparison_chart(df, 'int8', 'token_generation', '')
            if fig_token:
                st.plotly_chart(fig_token, use_container_width=True)
        
        with col3:
            fig_throughput = create_comparison_chart(df, 'int8', 'throughput', '')
            if fig_throughput:
                st.plotly_chart(fig_throughput, use_container_width=True)
    
    with tab2:
        st.subheader("FP16 Precision Comparisons")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_latency = create_comparison_chart(df, 'fp16', 'latency', '')
            if fig_latency:
                st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            fig_token = create_comparison_chart(df, 'fp16', 'token_generation', '')
            if fig_token:
                st.plotly_chart(fig_token, use_container_width=True)
        
        with col3:
            fig_throughput = create_comparison_chart(df, 'fp16', 'throughput', '')
            if fig_throughput:
                st.plotly_chart(fig_throughput, use_container_width=True)
    
    with tab3:
        st.subheader("FP32 Precision Comparisons")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_latency = create_comparison_chart(df, 'fp32', 'latency', '')
            if fig_latency:
                st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            fig_token = create_comparison_chart(df, 'fp32', 'token_generation', '')
            if fig_token:
                st.plotly_chart(fig_token, use_container_width=True)
        
        with col3:
            fig_throughput = create_comparison_chart(df, 'fp32', 'throughput', '')
            if fig_throughput:
                st.plotly_chart(fig_throughput, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed data table
    st.header("📋 Detailed Results")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        precision_filter = st.multiselect("Filter by Precision", df['precision'].unique(), default=df['precision'].unique())
    with col2:
        hardware_filter = st.multiselect("Filter by Hardware", df['hardware'].unique(), default=df['hardware'].unique())
    with col3:
        model_filter = st.multiselect("Filter by Model", df['model'].unique(), default=df['model'].unique())
    
    # Apply filters
    df_filtered = df[
        (df['precision'].isin(precision_filter)) &
        (df['hardware'].isin(hardware_filter)) &
        (df['model'].isin(model_filter))
    ]
    
    # Select columns to display
    display_columns = ['model', 'precision', 'hardware', 'latency_ms_mean', 'latency_ms_std', 
                      'tokens_per_second_mean', 'tokens_per_second_std', 
                      'throughput_mean', 'throughput_std', 'memory_used_mb_mean']
    
    df_display = df_filtered[display_columns].copy()
    
    # Rename columns for better display
    df_display.columns = ['Model', 'Precision', 'Hardware', 'Latency (ms)', 'Latency Std (ms)',
                         'Token Gen (tokens/s)', 'Token Gen Std (tokens/s)',
                         'Throughput (tokens/s)', 'Throughput Std (tokens/s)', 'Memory (MB)']
    
    st.dataframe(df_display, use_container_width=True)
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="benchmark_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
