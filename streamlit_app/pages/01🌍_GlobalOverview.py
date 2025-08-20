import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import folium
from streamlit_folium import folium_static

st.set_page_config(
    page_title="🌍 Global Overview",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load methane emissions data"""
    try:
        df = pd.read_csv('data/raw/Methane_final.csv')
        df.columns = df.columns.str.strip()
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['year'] = df['baseYear'].str.extract('(\d{4})').astype(float)
        return df
    except:
        # Sample data for demonstration
        np.random.seed(42)
        regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        countries = ['Algeria', 'Nigeria', 'Egypt', 'China', 'India', 'Japan', 'Germany', 
                    'France', 'UK', 'USA', 'Canada', 'Brazil', 'Argentina', 'Australia']
        types = ['Agriculture', 'Energy', 'Waste', 'Other']
        
        data = []
        for i in range(1000):
            data.append({
                'region': np.random.choice(regions),
                'country': np.random.choice(countries),
                'emissions': np.random.exponential(50) + np.random.normal(0, 10),
                'type': np.random.choice(types),
                'segment': np.random.choice(['Total', 'Bioenergy', 'Gas pipelines', 'Livestock']),
                'year': np.random.choice([2019, 2020, 2021, 2022])
            })
        
        return pd.DataFrame(data)

def create_world_map(df):
    """Create an interactive world map"""
    country_emissions = df.groupby('country')['emissions'].sum().reset_index()
    
    # Country coordinates (simplified)
    coordinates = {
        'Algeria': [28.0, 2.0], 'Nigeria': [9.0, 8.0], 'Egypt': [26.0, 30.0],
        'China': [35.0, 105.0], 'India': [20.0, 77.0], 'Japan': [36.0, 138.0],
        'Germany': [51.0, 9.0], 'France': [46.0, 2.0], 'UK': [54.0, -2.0],
        'USA': [40.0, -100.0], 'Canada': [60.0, -95.0],
        'Brazil': [-14.0, -51.0], 'Argentina': [-38.0, -64.0],
        'Australia': [-25.0, 133.0]
    }
    
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    
    for _, row in country_emissions.iterrows():
        if row['country'] in coordinates:
            lat, lon = coordinates[row['country']]
            folium.CircleMarker(
                location=[lat, lon],
                radius=min(row['emissions']/100, 50),
                popup=f"{row['country']}: {row['emissions']:.1f} Mt CO₂e",
                color='red',
                fill=True,
                fillOpacity=0.6
            ).add_to(m)
    
    return m

def main():
    st.title("🌍 Global Methane Emissions Overview")
    st.markdown("Comprehensive analysis of worldwide methane emissions patterns")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    selected_types = st.sidebar.multiselect(
        "Select Emission Types",
        options=df['type'].unique(),
        default=df['type'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['region'].isin(selected_regions)) &
        (df['type'].isin(selected_types))
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_emissions = filtered_df['emissions'].sum()
    avg_per_country = filtered_df.groupby('country')['emissions'].sum().mean()
    max_emitter = filtered_df.groupby('country')['emissions'].sum().idxmax()
    total_countries = filtered_df['country'].nunique()
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{total_emissions:,.0f}</h2>
            <p>Total Emissions (Mt CO₂e)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{avg_per_country:.1f}</h2>
            <p>Average per Country</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{max_emitter}</h2>
            <p>Largest Emitter</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{total_countries}</h2>
            <p>Countries Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # World map
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🗺️ Global Emissions Map")
    
    if len(filtered_df) > 0:
        world_map = create_world_map(filtered_df)
        folium_static(world_map, width=1200, height=500)
    else:
        st.warning("No data available for selected filters")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Regional comparison
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📊 Emissions by Region")
        
        region_data = filtered_df.groupby('region')['emissions'].sum().sort_values(ascending=True)
        
        fig_region = px.bar(
            x=region_data.values,
            y=region_data.index,
            orientation='h',
            color=region_data.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Emissions (Mt CO₂e)', 'y': 'Region'}
        )
        fig_region.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_region, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🏭 Sector Distribution")
        
        sector_data = filtered_df.groupby('type')['emissions'].sum()
        
        fig_pie = px.pie(
            values=sector_data.values,
            names=sector_data.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Country ranking
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🏆 Top 15 Emitting Countries")
    
    top_countries = filtered_df.groupby('country')['emissions'].sum().sort_values(ascending=False).head(15)
    
    fig_countries = px.bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        color=top_countries.values,
        color_continuous_scale='Reds',
        labels={'x': 'Emissions (Mt CO₂e)', 'y': 'Country'}
    )
    fig_countries.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_countries, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Time series if available
    if 'year' in filtered_df.columns and filtered_df['year'].notna().any():
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📈 Global Trends Over Time")
        
        yearly_regional = filtered_df.groupby(['year', 'region'])['emissions'].sum().reset_index()
        
        fig_trends = px.line(
            yearly_regional,
            x='year',
            y='emissions',
            color='region',
            markers=True,
            title="Regional Emissions Trends"
        )
        fig_trends.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Key insights
    st.markdown("""
    <div class="insight-box">
        <h3>🎯 Key Global Insights</h3>
        <ul>
            <li><strong>Regional Concentration:</strong> Major emissions concentrated in developing regions</li>
            <li><strong>Sector Dominance:</strong> Energy and Agriculture sectors lead global emissions</li>
            <li><strong>Growth Patterns:</strong> Emerging economies show rapid emission increases</li>
            <li><strong>Mitigation Opportunities:</strong> Technology transfer potential in high-emission regions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()