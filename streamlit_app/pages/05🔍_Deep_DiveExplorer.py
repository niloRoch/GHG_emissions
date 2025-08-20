import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns

st.set_page_config(
    page_title="🔍 Deep Dive Explorer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .explorer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .analysis-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .filter-panel {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stats-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .correlation-matrix {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load comprehensive emissions data for deep analysis"""
    try:
        df = pd.read_csv('data/raw/Methane_final.csv')
        df.columns = df.columns.str.strip()
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['year'] = df['baseYear'].str.extract('(\d{4})').astype(float)
        return df
    except:
        # Create rich sample data for deep analysis
        np.random.seed(42)
        
        # Enhanced data with more variables for correlation analysis
        regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        countries = ['China', 'USA', 'India', 'Brazil', 'Indonesia', 'Nigeria', 'Russia', 
                    'Mexico', 'Iran', 'Germany', 'Turkey', 'Canada', 'Australia', 'Algeria', 'Egypt']
        sectors = ['Agriculture', 'Energy', 'Waste', 'Other']
        subsectors = {
            'Agriculture': ['Livestock', 'Rice Cultivation', 'Agricultural Soils', 'Crop Residues', 'Manure Management'],
            'Energy': ['Oil & Gas', 'Coal Mining', 'Bioenergy', 'Power Generation', 'Refineries'],
            'Waste': ['Landfills', 'Wastewater', 'Composting', 'Incineration', 'Industrial Waste'],
            'Other': ['Industrial Processes', 'LULUCF', 'Solvent Use', 'Chemical Production', 'Other Sources']
        }
        
        data = []
        for i in range(2000):
            region = np.random.choice(regions)
            country = np.random.choice(countries)
            sector = np.random.choice(sectors)
            subsector = np.random.choice(subsectors[sector])
            year = np.random.choice(range(2015, 2023))
            
            # Create correlated variables for analysis
            base_emission = np.random.exponential(30)
            
            # Add regional and sectoral effects
            regional_multiplier = {'Asia': 1.5, 'Africa': 1.2, 'North America': 1.1, 
                                 'Europe': 0.8, 'South America': 1.0, 'Oceania': 0.6}[region]
            
            sectoral_multiplier = {'Agriculture': 1.3, 'Energy': 1.8, 'Waste': 0.9, 'Other': 0.7}[sector]
            
            emission = base_emission * regional_multiplier * sectoral_multiplier
            
            # Add synthetic variables for correlation analysis
            population_proxy = emission * np.random.uniform(0.5, 2.0)  # Correlated with emissions
            gdp_proxy = emission * np.random.uniform(0.3, 1.5) + np.random.normal(0, 10)  # Semi-correlated
            temperature_proxy = np.random.normal(25, 5) + emission * 0.01  # Weak correlation
            
            data.append({
                'region': region,
                'country': country,
                'type': sector,
                'segment': subsector,
                'year': year,
                'emissions': emission,
                'reason': np.random.choice(['Fugitive', 'Vented', 'Flared', 'All']),
                'population_proxy': population_proxy,
                'gdp_proxy': gdp_proxy,
                'temperature_proxy': temperature_proxy,
                'emission_intensity': emission / population_proxy if population_proxy > 0 else 0
            })
        
        return pd.DataFrame(data)

def create_correlation_analysis(df, numeric_cols):
    """Create correlation matrix and analysis"""
    corr_data = df[numeric_cols].corr()
    
    # Create correlation heatmap
    fig_corr = px.imshow(
        corr_data,
        title="Variable Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    fig_corr.update_layout(height=400)
    
    return fig_corr, corr_data

def perform_statistical_tests(df, groupby_col, value_col):
    """Perform statistical tests between groups"""
    groups = [group[value_col].dropna() for name, group in df.groupby(groupby_col)]
    
    if len(groups) < 2:
        return None
    
    # ANOVA test
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Descriptive statistics
        group_stats = df.groupby(groupby_col)[value_col].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(2)
        
        return {
            'anova_f': f_stat,
            'anova_p': p_value,
            'group_stats': group_stats,
            'significant': p_value < 0.05
        }
    except:
        return None

def create_advanced_visualizations(df, x_col, y_col, color_col=None, size_col=None):
    """Create advanced scatter plot with multiple dimensions"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=[col for col in df.columns if col in ['country', 'region', 'type', 'year']],
        title=f"{y_col} vs {x_col}",
        opacity=0.7
    )
    
    # Add trendline
    if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
        fig.add_traces(px.scatter(df, x=x_col, y=y_col, trendline="ols", opacity=0).data[1:])
    
    fig.update_layout(height=500)
    return fig

def main():
    st.title("🔍 Deep Dive Data Explorer")
    st.markdown("Advanced analytics, custom filtering, and statistical analysis")
    
    # Load data
    df = load_data()
    
    # Data overview
    st.markdown("<div class='explorer-card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}", "Data Points")
    
    with col2:
        st.metric("Variables", f"{len(df.columns)}", "Dimensions")
    
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}", "Quality Check")
    
    with col4:
        st.metric("Time Span", f"{df['year'].nunique() if 'year' in df.columns else 'N/A'}", "Years")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Advanced Filtering Panel
    st.sidebar.header("🎛️ Advanced Filters")
    
    # Dynamic filter creation
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Multi-level filtering
    filters = {}
    for col in categorical_cols[:6]:  # Limit to first 6 categorical columns
        if df[col].nunique() <= 50:  # Only show if not too many unique values
            filters[col] = st.sidebar.multiselect(
                f"Filter {col.title()}",
                options=df[col].unique(),
                default=df[col].unique()[:10] if df[col].nunique() > 10 else df[col].unique()
            )
    
    # Numerical filters
    for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
        if col != 'year':  # Skip year for now
            min_val, max_val = float(df[col].min()), float(df[col].max())
            filters[col] = st.sidebar.slider(
                f"{col.title()} Range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
    
    # Apply filters
    filtered_df = df.copy()
    for col, values in filters.items():
        if col in categorical_cols:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        else:  # numerical
            filtered_df = filtered_df[
                (filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])
            ]
    
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")
    
    # Analysis Type Selection
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Correlation Analysis", "Statistical Testing", "Distribution Analysis", 
         "Advanced Scatter Analysis", "Custom Visualization", "Data Quality Check"]
    )
    
    if analysis_type == "Correlation Analysis":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("📊 Correlation Analysis")
        
        # Select variables for correlation
        selected_vars = st.multiselect(
            "Select variables for correlation analysis",
            options=numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols
        )
        
        if len(selected_vars) >= 2:
            fig_corr, corr_matrix = create_correlation_analysis(filtered_df, selected_vars)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Show strongest correlations
            st.subheader("🔗 Strongest Correlations")
            
            # Create correlation pairs
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_pairs_df.head(10), use_container_width=True)
        
        else:
            st.warning("Please select at least 2 numerical variables for correlation analysis.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Statistical Testing":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("📈 Statistical Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            groupby_var = st.selectbox("Group by Variable", categorical_cols)
        
        with col2:
            test_var = st.selectbox("Test Variable", numerical_cols)
        
        if groupby_var and test_var:
            test_results = perform_statistical_tests(filtered_df, groupby_var, test_var)
            
            if test_results:
                # Display ANOVA results
                st.markdown(f"""
                <div class="stats-box">
                    <h4>ANOVA Results</h4>
                    <p>F-statistic: {test_results['anova_f']:.3f}</p>
                    <p>P-value: {test_results['anova_p']:.6f}</p>
                    <p>Significant: {'Yes' if test_results['significant'] else 'No'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Group statistics
                st.subheader("📊 Group Statistics")
                st.dataframe(test_results['group_stats'], use_container_width=True)
                
                # Box plot
                fig_box = px.box(
                    filtered_df,
                    x=groupby_var,
                    y=test_var,
                    title=f"{test_var} Distribution by {groupby_var}"
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            
            else:
                st.error("Unable to perform statistical tests on selected variables.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Distribution Analysis":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("📊 Distribution Analysis")
        
        selected_var = st.selectbox("Select Variable for Distribution Analysis", numerical_cols)
        
        if selected_var:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    filtered_df,
                    x=selected_var,
                    nbins=30,
                    title=f"Distribution of {selected_var}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot and statistics
                fig_box = px.box(
                    filtered_df,
                    y=selected_var,
                    title=f"Box Plot of {selected_var}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Distribution statistics
            stats_data = filtered_df[selected_var].describe()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Central Tendency</h4>
                    <p>Mean: {stats_data['mean']:.2f}</p>
                    <p>Median: {stats_data['50%']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Variability</h4>
                    <p>Std Dev: {stats_data['std']:.2f}</p>
                    <p>Range: {stats_data['max'] - stats_data['min']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                skewness = stats.skew(filtered_df[selected_var].dropna())
                kurtosis = stats.kurtosis(filtered_df[selected_var].dropna())
                
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Shape</h4>
                    <p>Skewness: {skewness:.2f}</p>
                    <p>Kurtosis: {kurtosis:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Advanced Scatter Analysis":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("🎯 Advanced Scatter Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_var = st.selectbox("X-axis Variable", numerical_cols, key="x_scatter")
        
        with col2:
            y_var = st.selectbox("Y-axis Variable", numerical_cols, key="y_scatter")
        
        with col3:
            color_var = st.selectbox("Color Variable", ['None'] + categorical_cols, key="color_scatter")
            color_var = None if color_var == 'None' else color_var
        
        with col4:
            size_var = st.selectbox("Size Variable", ['None'] + numerical_cols, key="size_scatter")
            size_var = None if size_var == 'None' else size_var
        
        if x_var and y_var:
            fig_scatter = create_advanced_visualizations(
                filtered_df, x_var, y_var, color_var, size_var
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Calculate correlation
            if x_var != y_var:
                correlation = filtered_df[x_var].corr(filtered_df[y_var])
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Correlation Coefficient</h4>
                    <p>{correlation:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Custom Visualization":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("🎨 Custom Visualization Builder")
        
        viz_type = st.selectbox(
            "Visualization Type",
            ["Bar Chart", "Line Chart", "Heatmap", "Treemap", "Sunburst", "3D Scatter"]
        )
        
        if viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", categorical_cols + numerical_cols)
            with col2:
                y_var = st.selectbox("Y Variable", numerical_cols)
            
            if x_var and y_var:
                agg_data = filtered_df.groupby(x_var)[y_var].mean().sort_values(ascending=False)
                fig = px.bar(x=agg_data.index, y=agg_data.values, title=f"Average {y_var} by {x_var}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap":
            pivot_vars = st.multiselect("Select variables for heatmap", categorical_cols, max_selections=3)
            value_var = st.selectbox("Value variable", numerical_cols)
            
            if len(pivot_vars) == 2 and value_var:
                pivot_data = filtered_df.pivot_table(
                    values=value_var, 
                    index=pivot_vars[0], 
                    columns=pivot_vars[1], 
                    aggfunc='mean'
                )
                fig = px.imshow(pivot_data, title=f"Heatmap: {value_var}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Treemap":
            hierarchy = st.multiselect("Select hierarchy levels", categorical_cols, max_selections=3)
            value_var = st.selectbox("Value variable", numerical_cols, key="treemap_value")
            
            if hierarchy and value_var:
                fig = px.treemap(filtered_df, path=hierarchy, values=value_var, title="Treemap Visualization")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Data Quality Check":
        st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Missing Values Analysis")
            missing_data = filtered_df.isnull().sum()
            missing_pct = (missing_data / len(filtered_df) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            }).sort_values('Missing Count', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.markdown("### Data Types & Unique Values")
            
            dtype_info = []
            for col in filtered_df.columns:
                dtype_info.append({
                    'Column': col,
                    'Data Type': str(filtered_df[col].dtype),
                    'Unique Values': filtered_df[col].nunique(),
                    'Sample Value': str(filtered_df[col].iloc[0]) if len(filtered_df) > 0 else 'N/A'
                })
            
            dtype_df = pd.DataFrame(dtype_info)
            st.dataframe(dtype_df, use_container_width=True)
        
        # Outlier detection for numerical columns
        st.markdown("### Outlier Detection")
        
        numerical_subset = filtered_df.select_dtypes(include=[np.number])
        if len(numerical_subset.columns) > 0:
            outlier_col = st.selectbox("Select column for outlier analysis", numerical_subset.columns)
            
            if outlier_col:
                Q1 = filtered_df[outlier_col].quantile(0.25)
                Q3 = filtered_df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = filtered_df[
                    (filtered_df[outlier_col] < lower_bound) | 
                    (filtered_df[outlier_col] > upper_bound)
                ]
                
                st.markdown(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(filtered_df)*100:.1f}%)")
                
                if len(outliers) > 0:
                    fig_outlier = px.box(filtered_df, y=outlier_col, title=f"Outlier Detection: {outlier_col}")
                    st.plotly_chart(fig_outlier, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data Export
    st.markdown("### 💾 Export Filtered Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV file",
                data=csv,
                file_name='filtered_emissions_data.csv',
                mime='text/csv'
            )
    
    with col2:
        if st.button("📈 Download Excel"):
            # Note: This would require openpyxl in requirements
            st.info("Excel download would be available with openpyxl installed")
    
    with col3:
        if st.button("📋 Show Summary"):
            st.json({
                'Total Records': len(filtered_df),
                'Date Range': f"{filtered_df['year'].min()}-{filtered_df['year'].max()}" if 'year' in filtered_df.columns else 'N/A',
                'Total Emissions': f"{filtered_df['emissions'].sum():.1f}" if 'emissions' in filtered_df.columns else 'N/A',
                'Countries': filtered_df['country'].nunique() if 'country' in filtered_df.columns else 'N/A',
                'Regions': filtered_df['region'].nunique() if 'region' in filtered_df.columns else 'N/A'
            })

if __name__ == "__main__":
    main()