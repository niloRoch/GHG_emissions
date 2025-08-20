import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="🗺️ Geographic Insights",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .geo-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .map-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .cluster-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process geographic emissions data"""
    try:
        df = pd.read_csv('data/raw/Methane_final.csv')
        df.columns = df.columns.str.strip()
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['year'] = df['baseYear'].str.extract('(\d{4})').astype(float)
        return df
    except:
        # Enhanced sample data with geographic focus
        np.random.seed(42)
        
        # More realistic country data
        countries_data = {
            'China': {'region': 'Asia', 'lat': 35.0, 'lon': 105.0, 'pop': 1400},
            'India': {'region': 'Asia', 'lat': 20.0, 'lon': 77.0, 'pop': 1380},
            'USA': {'region': 'North America', 'lat': 40.0, 'lon': -100.0, 'pop': 330},
            'Indonesia': {'region': 'Asia', 'lat': -5.0, 'lon': 120.0, 'pop': 270},
            'Brazil': {'region': 'South America', 'lat': -14.0, 'lon': -51.0, 'pop': 215},
            'Nigeria': {'region': 'Africa', 'lat': 9.0, 'lon': 8.0, 'pop': 220},
            'Russia': {'region': 'Europe', 'lat': 60.0, 'lon': 100.0, 'pop': 145},
            'Mexico': {'region': 'North America', 'lat': 23.0, 'lon': -102.0, 'pop': 130},
            'Iran': {'region': 'Asia', 'lat': 32.0, 'lon': 53.0, 'pop': 85},
            'Germany': {'region': 'Europe', 'lat': 51.0, 'lon': 9.0, 'pop': 83},
            'Turkey': {'region': 'Europe', 'lat': 39.0, 'lon': 35.0, 'pop': 85},
            'Canada': {'region': 'North America', 'lat': 60.0, 'lon': -95.0, 'pop': 38},
            'Australia': {'region': 'Oceania', 'lat': -25.0, 'lon': 133.0, 'pop': 26},
            'Argentina': {'region': 'South America', 'lat': -38.0, 'lon': -64.0, 'pop': 45},
            'Algeria': {'region': 'Africa', 'lat': 28.0, 'lon': 2.0, 'pop': 45}
        }
        
        types = ['Agriculture', 'Energy', 'Waste', 'Other']
        segments = ['Livestock', 'Oil & Gas', 'Landfills', 'Rice Cultivation', 'Coal Mining']
        
        data = []
        for country, info in countries_data.items():
            # Generate realistic emissions based on population and region
            base_emission = info['pop'] / 10 + np.random.normal(0, 20)
            
            for _ in range(20):  # Multiple entries per country
                sector = np.random.choice(types)
                # Adjust emissions based on sector and region
                sector_multiplier = {'Agriculture': 1.5, 'Energy': 2.0, 'Waste': 0.8, 'Other': 0.5}[sector]
                
                data.append({
                    'region': info['region'],
                    'country': country,
                    'emissions': max(0, base_emission * sector_multiplier * np.random.uniform(0.5, 1.5)),
                    'type': sector,
                    'segment': np.random.choice(segments),
                    'year': np.random.choice([2019, 2020, 2021, 2022]),
                    'latitude': info['lat'] + np.random.normal(0, 2),
                    'longitude': info['lon'] + np.random.normal(0, 2),
                    'population': info['pop']
                })
        
        return pd.DataFrame(data)

def create_choropleth_map(df):
    """Create a choropleth map showing emissions by country"""
    country_emissions = df.groupby('country')['emissions'].sum().reset_index()
    
    fig = px.choropleth(
        country_emissions,
        locations='country',
        locationmode='country names',
        color='emissions',
        hover_name='country',
        hover_data={'emissions': ':,.1f'},
        color_continuous_scale='Reds',
        title='Global Methane Emissions by Country'
    )
    
    fig.update_layout(
        height=500,
        geo=dict(showframe=False, showcoastlines=True)
    )
    
    return fig

def create_bubble_map(df):
    """Create bubble map with emissions and additional metrics"""
    country_data = df.groupby('country').agg({
        'emissions': 'sum',
        'latitude': 'first',
        'longitude': 'first',
        'population': 'first',
        'region': 'first'
    }).reset_index()
    
    # Calculate emissions per capita
    country_data['emissions_per_capita'] = country_data['emissions'] / country_data['population']
    
    fig = px.scatter_geo(
        country_data,
        lat='latitude',
        lon='longitude',
        size='emissions',
        color='emissions_per_capita',
        hover_name='country',
        hover_data={
            'emissions': ':,.1f',
            'emissions_per_capita': ':.2f',
            'population': ':,',
            'region': True
        },
        color_continuous_scale='Viridis',
        title='Emissions by Country (Size: Total, Color: Per Capita)'
    )
    
    fig.update_layout(height=500)
    return fig

def perform_country_clustering(df):
    """Perform k-means clustering on countries based on emission patterns"""
    # Prepare data for clustering
    country_sector = df.pivot_table(
        values='emissions', 
        index='country', 
        columns='type', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Add population and regional data
    country_info = df.groupby('country').agg({
        'population': 'first',
        'region': 'first'
    })
    
    clustering_data = country_sector.join(country_info)
    
    # Standardize the data
    scaler = StandardScaler()
    features = clustering_data[['Agriculture', 'Energy', 'Waste', 'Other', 'population']]
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clustering_data['cluster'] = kmeans.fit_predict(features_scaled)
    
    return clustering_data, kmeans

def main():
    st.title("🗺️ Geographic Analysis of Methane Emissions")
    st.markdown("Spatial patterns, regional insights, and country-level analysis")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Geographic Controls")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Country Overview", "Regional Comparison", "Clustering Analysis", "Hotspot Detection"]
    )
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Filter data
    filtered_df = df[df['region'].isin(selected_regions)]
    
    # Geographic metrics
    st.markdown("<div class='geo-card'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    total_countries = filtered_df['country'].nunique()
    total_regions = filtered_df['region'].nunique()
    top_emitter = filtered_df.groupby('country')['emissions'].sum().idxmax()
    regional_spread = filtered_df.groupby('region')['emissions'].sum().std()
    
    with col1:
        st.metric("Countries", total_countries, "Global Coverage")
    
    with col2:
        st.metric("Regions", total_regions, "Continental Scope")
    
    with col3:
        st.metric("Top Emitter", top_emitter, "Leading Country")
    
    with col4:
        st.metric("Regional Spread", f"{regional_spread:.0f}", "Variability Index")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main analysis based on selection
    if analysis_type == "Country Overview":
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("<div class='map-container'>", unsafe_allow_html=True)
            st.subheader("🌍 Global Emissions Choropleth")
            
            choropleth_fig = create_choropleth_map(filtered_df)
            st.plotly_chart(choropleth_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_right:
            st.markdown("<div class='map-container'>", unsafe_allow_html=True)
            st.subheader("🏆 Top 10 Countries")
            
            top_countries = filtered_df.groupby('country')['emissions'].sum().sort_values(ascending=False).head(10)
            
            for i, (country, emissions) in enumerate(top_countries.items(), 1):
                st.markdown(f"**{i}. {country}**")
                st.markdown(f"{emissions:.1f} Mt CO₂e")
                st.progress(emissions / top_countries.max())
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Bubble map
        st.markdown("<div class='map-container'>", unsafe_allow_html=True)
        st.subheader("💫 Emissions vs Population Analysis")
        
        if 'population' in filtered_df.columns:
            bubble_fig = create_bubble_map(filtered_df)
            st.plotly_chart(bubble_fig, use_container_width=True)
        else:
            st.info("Population data not available for bubble map analysis")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Regional Comparison":
        st.markdown("<div class='map-container'>", unsafe_allow_html=True)
        st.subheader("🌎 Regional Comparison Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional totals
            regional_totals = filtered_df.groupby('region')['emissions'].sum().sort_values(ascending=True)
            
            fig_regional = px.bar(
                x=regional_totals.values,
                y=regional_totals.index,
                orientation='h',
                color=regional_totals.values,
                color_continuous_scale='Blues',
                title="Total Emissions by Region"
            )
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            # Regional sector breakdown
            regional_sector = filtered_df.groupby(['region', 'type'])['emissions'].sum().reset_index()
            
            fig_sector = px.treemap(
                regional_sector,
                path=['region', 'type'],
                values='emissions',
                title="Regional Sector Breakdown"
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Regional trends
        if 'year' in filtered_df.columns and filtered_df['year'].notna().any():
            regional_trends = filtered_df.groupby(['year', 'region'])['emissions'].sum().reset_index()
            
            fig_trends = px.line(
                regional_trends,
                x='year',
                y='emissions',
                color='region',
                markers=True,
                title="Regional Emission Trends Over Time"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Clustering Analysis":
        st.markdown("<div class='map-container'>", unsafe_allow_html=True)
        st.subheader("🎯 Country Clustering Analysis")
        
        if len(filtered_df['country'].unique()) >= 4:
            clustering_data, kmeans = perform_country_clustering(filtered_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot of clusters
                fig_cluster = px.scatter(
                    clustering_data.reset_index(),
                    x='Energy',
                    y='Agriculture',
                    color='cluster',
                    size='population',
                    hover_name='country',
                    title="Country Clusters (Energy vs Agriculture)"
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with col2:
                # Cluster characteristics
                st.markdown("### Cluster Characteristics")
                
                for cluster_id in sorted(clustering_data['cluster'].unique()):
                    cluster_countries = clustering_data[clustering_data['cluster'] == cluster_id]
                    avg_emissions = cluster_countries[['Agriculture', 'Energy', 'Waste', 'Other']].mean()
                    
                    st.markdown(f"""
                    <div class="cluster-info">
                        <strong>Cluster {cluster_id}</strong> ({len(cluster_countries)} countries)<br>
                        Main characteristic: {avg_emissions.idxmax()}-dominant<br>
                        Countries: {', '.join(cluster_countries.index[:3].tolist())}{'...' if len(cluster_countries) > 3 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Cluster composition
            cluster_composition = clustering_data.groupby('cluster').agg({
                'Agriculture': 'mean',
                'Energy': 'mean', 
                'Waste': 'mean',
                'Other': 'mean'
            })
            
            fig_composition = px.bar(
                cluster_composition.T,
                title="Average Sector Composition by Cluster",
                labels={'index': 'Sector', 'value': 'Average Emissions'}
            )
            st.plotly_chart(fig_composition, use_container_width=True)
        
        else:
            st.warning("Need at least 4 countries for clustering analysis")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif analysis_type == "Hotspot Detection":
        st.markdown("<div class='map-container'>", unsafe_allow_html=True)
        st.subheader("🔥 Emission Hotspot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Identify hotspots using quantiles
            country_emissions = filtered_df.groupby('country')['emissions'].sum()
            hotspot_threshold = country_emissions.quantile(0.8)  # Top 20%
            
            hotspots = country_emissions[country_emissions >= hotspot_threshold].sort_values(ascending=False)
            
            st.markdown("### 🚨 High Emission Countries")
            for country, emissions in hotspots.items():
                pct_of_total = (emissions / country_emissions.sum()) * 100
                st.markdown(f"**{country}**: {emissions:.1f} Mt ({pct_of_total:.1f}%)")
                st.progress(emissions / hotspots.max())
        
        with col2:
            # Hotspot intensity by sector
            hotspot_countries = hotspots.index.tolist()
            hotspot_data = filtered_df[filtered_df['country'].isin(hotspot_countries)]
            
            hotspot_sectors = hotspot_data.groupby(['country', 'type'])['emissions'].sum().reset_index()
            
            fig_hotspot = px.bar(
                hotspot_sectors,
                x='country',
                y='emissions',
                color='type',
                title="Hotspot Composition by Sector"
            )
            fig_hotspot.update_xaxes(tickangle=45)
            st.plotly_chart(fig_hotspot, use_container_width=True)
        
        # Hotspot geographic distribution
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            hotspot_geo = filtered_df[filtered_df['country'].isin(hotspot_countries)]
            hotspot_summary = hotspot_geo.groupby('country').agg({
                'emissions': 'sum',
                'latitude': 'first',
                'longitude': 'first'
            }).reset_index()
            
            fig_geo_hotspot = px.scatter_geo(
                hotspot_summary,
                lat='latitude',
                lon='longitude',
                size='emissions',
                hover_name='country',
                title="Geographic Distribution of Hotspots",
                color_discrete_sequence=['red']
            )
            st.plotly_chart(fig_geo_hotspot, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparative analysis
    st.markdown("<div class='map-container'>", unsafe_allow_html=True)
    st.subheader("⚖️ Comparative Geographic Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Emissions density by region
        regional_density = filtered_df.groupby('region').agg({
            'emissions': 'sum',
            'country': 'nunique'
        })
        regional_density['density'] = regional_density['emissions'] / regional_density['country']
        regional_density = regional_density.sort_values('density', ascending=False)
        
        fig_density = px.bar(
            x=regional_density.index,
            y=regional_density['density'],
            title="Emission Density (Mt/Country)",
            color=regional_density['density'],
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_density, use_container_width=True)
    
    with col2:
        # Regional diversity index
        regional_diversity = filtered_df.groupby('region')['type'].nunique().sort_values(ascending=False)
        
        fig_diversity = px.bar(
            x=regional_diversity.index,
            y=regional_diversity.values,
            title="Sectoral Diversity by Region",
            color=regional_diversity.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    with col3:
        # Top country per region
        regional_tops = filtered_df.groupby(['region', 'country'])['emissions'].sum().reset_index()
        regional_tops = regional_tops.loc[regional_tops.groupby('region')['emissions'].idxmax()]
        
        fig_tops = px.bar(
            regional_tops,
            x='region',
            y='emissions',
            color='country',
            title="Top Emitter per Region"
        )
        st.plotly_chart(fig_tops, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key geographic insights
    top_region = filtered_df.groupby('region')['emissions'].sum().idxmax()
    most_diverse_region = filtered_df.groupby('region')['type'].nunique().idxmax()
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>🎯 Key Geographic Insights</h3>
        <ul>
            <li><strong>Regional Leader:</strong> {top_region} dominates global methane emissions</li>
            <li><strong>Sectoral Diversity:</strong> {most_diverse_region} shows highest sectoral diversity</li>
            <li><strong>Concentration Pattern:</strong> Emissions highly concentrated in few countries</li>
            <li><strong>Development Correlation:</strong> Higher emissions often correlate with industrial development</li>
            <li><strong>Mitigation Priority:</strong> Focus on top 20% emitters could yield 80% impact</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()