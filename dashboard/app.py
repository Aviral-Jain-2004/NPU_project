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
    st.title("🚀 NPU LLM Benchmark Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No benchmark data found. Please run the benchmark script first.")
        st.info("Run `python scripts/run_benchmarks.py` on your remote desktop to generate benchmark data.")
        return
    
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
