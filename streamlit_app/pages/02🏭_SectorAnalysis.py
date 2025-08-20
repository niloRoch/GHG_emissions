import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="🏭 Sector Analysis",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .sector-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .analysis-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
    .insight-highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process methane emissions data"""
    try:
        df = pd.read_csv('data/raw/Methane_final.csv')
        df.columns = df.columns.str.strip()
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['year'] = df['baseYear'].str.extract('(\d{4})').astype(float)
        return df
    except:
        # Enhanced sample data with more sector details
        np.random.seed(42)
        sectors_detailed = {
            'Agriculture': ['Livestock', 'Rice Cultivation', 'Agricultural Soils', 'Crop Residues'],
            'Energy': ['Oil & Gas', 'Coal Mining', 'Bioenergy', 'Power Generation'],
            'Waste': ['Landfills', 'Wastewater', 'Composting', 'Incineration'],
            'Other': ['Industrial Processes', 'LULUCF', 'Solvent Use', 'Other Sources']
        }
        
        regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        countries = ['Algeria', 'Nigeria', 'Egypt', 'China', 'India', 'Japan', 'Germany', 
                    'France', 'UK', 'USA', 'Canada', 'Brazil', 'Argentina', 'Australia']
        
        data = []
        for i in range(1200):
            sector = np.random.choice(list(sectors_detailed.keys()))
            subsector = np.random.choice(sectors_detailed[sector])
            
            data.append({
                'region': np.random.choice(regions),
                'country': np.random.choice(countries),
                'emissions': np.random.exponential(30) + np.random.normal(0, 5),
                'type': sector,
                'segment': subsector,
                'year': np.random.choice([2019, 2020, 2021, 2022]),
                'reason': np.random.choice(['All', 'Fugitive', 'Vented', 'Flared'])
            })
        
        return pd.DataFrame(data)

def create_sector_comparison(df):
    """Create comprehensive sector comparison chart"""
    sector_stats = df.groupby('type').agg({
        'emissions': ['sum', 'mean', 'std', 'count']
    }).round(2)
    sector_stats.columns = ['Total', 'Mean', 'StdDev', 'Count']
    sector_stats = sector_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Total Emissions by Sector', 'Average Emissions', 
                       'Emission Distribution', 'Data Points Count'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}]]
    )
    
    # Total emissions
    fig.add_trace(
        go.Bar(x=sector_stats['type'], y=sector_stats['Total'], 
               name='Total', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Average emissions
    fig.add_trace(
        go.Bar(x=sector_stats['type'], y=sector_stats['Mean'], 
               name='Average', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Box plot for distribution
    for sector in df['type'].unique():
        sector_data = df[df['type'] == sector]['emissions']
        fig.add_trace(
            go.Box(y=sector_data, name=sector, showlegend=False),
            row=2, col=1
        )
    
    # Count of data points
    fig.add_trace(
        go.Bar(x=sector_stats['type'], y=sector_stats['Count'], 
               name='Count', marker_color='lightcoral'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_subsector_analysis(df, selected_sector):
    """Analyze subsectors within a selected sector"""
    sector_df = df[df['type'] == selected_sector]
    subsector_data = sector_df.groupby('segment')['emissions'].sum().sort_values(ascending=False)
    
    # Create sunburst chart
    fig_sunburst = px.sunburst(
        sector_df.groupby(['type', 'segment'])['emissions'].sum().reset_index(),
        path=['type', 'segment'],
        values='emissions',
        title=f'Subsector Breakdown - {selected_sector}'
    )
    
    return fig_sunburst, subsector_data

def main():
    st.title("🏭 Methane Emissions Sector Analysis")
    st.markdown("Deep dive into sectoral emissions patterns and trends")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Sector selection
    selected_sector = st.sidebar.selectbox(
        "Focus Sector",
        options=['All'] + list(df['type'].unique()),
        index=0
    )
    
    # Region filter
    selected_regions = st.sidebar.multiselect(
        "Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Filter data
    filtered_df = df[df['region'].isin(selected_regions)]
    
    # Overview metrics
    st.markdown("<div class='sector-card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    total_sectors = filtered_df['type'].nunique()
    dominant_sector = filtered_df.groupby('type')['emissions'].sum().idxmax()
    total_emissions = filtered_df['emissions'].sum()
    sector_diversity = filtered_df.groupby('type')['emissions'].sum().std() / filtered_df.groupby('type')['emissions'].sum().mean()
    
    with col1:
        st.metric("Total Sectors", total_sectors, "Complete Coverage")
    
    with col2:
        st.metric("Dominant Sector", dominant_sector, "Highest Emissions")
    
    with col3:
        st.metric("Total Emissions", f"{total_emissions:,.0f} Mt", "CO₂ Equivalent")
    
    with col4:
        st.metric("Sector Diversity", f"{sector_diversity:.2f}", "Coefficient of Variation")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main analysis
    if selected_sector == 'All':
        # Comprehensive sector comparison
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        st.subheader("📊 Comprehensive Sector Comparison")
        
        sector_fig = create_sector_comparison(filtered_df)
        st.plotly_chart(sector_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sector trends over time
        if 'year' in filtered_df.columns and filtered_df['year'].notna().any():
            st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
            st.subheader("📈 Sectoral Trends Over Time")
            
            yearly_sector = filtered_df.groupby(['year', 'type'])['emissions'].sum().reset_index()
            
            fig_trends = px.area(
                yearly_sector,
                x='year',
                y='emissions',
                color='type',
                title="Sectoral Emissions Evolution"
            )
            fig_trends.update_layout(height=400)
            st.plotly_chart(fig_trends, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Focused sector analysis
        sector_df = filtered_df[filtered_df['type'] == selected_sector]
        
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        st.subheader(f"🔍 {selected_sector} Sector Deep Dive")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Subsector analysis
            sunburst_fig, subsector_data = create_subsector_analysis(filtered_df, selected_sector)
            st.plotly_chart(sunburst_fig, use_container_width=True)
        
        with col_right:
            st.markdown(f"### Top {selected_sector} Sources")
            for i, (subsector, emissions) in enumerate(subsector_data.head(5).items(), 1):
                st.markdown(f"**{i}. {subsector}**")
                st.markdown(f"{emissions:.1f} Mt CO₂e")
                st.progress(emissions / subsector_data.max())
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Regional breakdown for selected sector
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        st.subheader(f"🌍 {selected_sector} by Region")
        
        regional_sector = sector_df.groupby('region')['emissions'].sum().sort_values(ascending=True)
        
        fig_regional = px.bar(
            x=regional_sector.values,
            y=regional_sector.index,
            orientation='h',
            color=regional_sector.values,
            color_continuous_scale='Viridis'
        )
        fig_regional.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_regional, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Cross-sector analysis
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.subheader("🔄 Cross-Sector Analysis")
    
    # Create correlation matrix
    sector_pivot = filtered_df.pivot_table(
        values='emissions', 
        index='country', 
        columns='type', 
        aggfunc='sum',
        fill_value=0
    )
    
    if len(sector_pivot.columns) > 1:
        correlation_matrix = sector_pivot.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Sector Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Efficiency analysis
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.subheader("⚡ Sector Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Emissions intensity by sector
        sector_intensity = filtered_df.groupby('type').agg({
            'emissions': ['sum', 'count']
        })
        sector_intensity.columns = ['Total', 'Count']
        sector_intensity['Intensity'] = sector_intensity['Total'] / sector_intensity['Count']
        sector_intensity = sector_intensity.sort_values('Intensity', ascending=False)
        
        fig_intensity = px.bar(
            x=sector_intensity.index,
            y=sector_intensity['Intensity'],
            title="Emission Intensity by Sector",
            color=sector_intensity['Intensity'],
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_intensity, use_container_width=True)
    
    with col2:
        # Sector contribution vs efficiency
        sector_contrib = filtered_df.groupby('type')['emissions'].sum()
        sector_contrib_pct = (sector_contrib / sector_contrib.sum() * 100).round(1)
        
        fig_contrib = px.pie(
            values=sector_contrib_pct.values,
            names=sector_contrib_pct.index,
            title="Sector Contribution %"
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key insights
    top_sector = filtered_df.groupby('type')['emissions'].sum().idxmax()
    top_sector_pct = (filtered_df.groupby('type')['emissions'].sum().max() / filtered_df['emissions'].sum() * 100).round(1)
    
    st.markdown(f"""
    <div class="insight-highlight">
        <h3>🎯 Key Sectoral Insights</h3>
        <ul>
            <li><strong>{top_sector}</strong> dominates with {top_sector_pct}% of total emissions</li>
            <li><strong>Regional Variation:</strong> Different sectors dominate in different regions</li>
            <li><strong>Subsector Opportunities:</strong> Targeted interventions can yield significant reductions</li>
            <li><strong>Cross-Sector Synergies:</strong> Integrated approaches may be more effective</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()