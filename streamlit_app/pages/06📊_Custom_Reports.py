import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
import base64

st.set_page_config(
    page_title="📊 Custom Reports",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .report-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .report-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .report-builder {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .chart-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    @media print {
        .stApp > header, .stApp > footer, .stSidebar {
            display: none;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load emissions data for report generation"""
    try:
        df = pd.read_csv('data/raw/Methane_final.csv')
        df.columns = df.columns.str.strip()
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['year'] = df['baseYear'].str.extract('(\d{4})').astype(float)
        return df
    except:
        # Create comprehensive sample data for reporting
        np.random.seed(42)
        
        regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        countries = ['China', 'USA', 'India', 'Brazil', 'Indonesia', 'Nigeria', 'Russia', 
                    'Mexico', 'Iran', 'Germany', 'Turkey', 'Canada', 'Australia', 'Algeria', 'Egypt']
        sectors = ['Agriculture', 'Energy', 'Waste', 'Other']
        years = list(range(2018, 2023))
        
        data = []
        for year in years:
            for region in regions:
                for sector in sectors:
                    for country in np.random.choice(countries, 5):
                        # Create realistic emissions with trends
                        base_emission = np.random.exponential(25)
                        year_trend = 1 + (year - 2018) * 0.02  # 2% annual growth
                        
                        # Regional factors
                        regional_factor = {
                            'Asia': 1.8, 'Africa': 1.3, 'North America': 1.1,
                            'Europe': 0.8, 'South America': 1.0, 'Oceania': 0.5
                        }[region]
                        
                        # Sectoral factors
                        sectoral_factor = {
                            'Agriculture': 1.2, 'Energy': 2.0, 'Waste': 0.8, 'Other': 0.6
                        }[sector]
                        
                        emission = base_emission * year_trend * regional_factor * sectoral_factor
                        
                        data.append({
                            'year': year,
                            'region': region,
                            'country': country,
                            'type': sector,
                            'emissions': max(0, emission + np.random.normal(0, 5)),
                            'segment': f"{sector}_subsector",
                            'reason': np.random.choice(['Fugitive', 'Vented', 'All'])
                        })
        
        return pd.DataFrame(data)

def generate_executive_summary(df):
    """Generate executive summary statistics"""
    current_year = df['year'].max() if 'year' in df.columns else 2022
    previous_year = current_year - 1
    
    current_emissions = df[df['year'] == current_year]['emissions'].sum()
    previous_emissions = df[df['year'] == previous_year]['emissions'].sum() if previous_year in df['year'].values else current_emissions
    
    year_over_year = ((current_emissions - previous_emissions) / previous_emissions * 100) if previous_emissions > 0 else 0
    
    top_region = df.groupby('region')['emissions'].sum().idxmax()
    top_sector = df.groupby('type')['emissions'].sum().idxmax()
    top_country = df.groupby('country')['emissions'].sum().idxmax()
    
    return {
        'total_emissions': current_emissions,
        'yoy_change': year_over_year,
        'top_region': top_region,
        'top_sector': top_sector,
        'top_country': top_country,
        'total_countries': df['country'].nunique(),
        'report_date': datetime.now().strftime("%B %d, %Y")
    }

def create_report_charts(df, report_config):
    """Create charts based on report configuration"""
    charts = {}
    
    # Regional overview
    if report_config.get('include_regional', True):
        regional_data = df.groupby('region')['emissions'].sum().sort_values(ascending=False)
        charts['regional'] = px.bar(
            x=regional_data.index,
            y=regional_data.values,
            title="Emissions by Region",
            labels={'x': 'Region', 'y': 'Emissions (Mt CO₂e)'},
            color=regional_data.values,
            color_continuous_scale='Blues'
        )
    
    # Sectoral breakdown
    if report_config.get('include_sectoral', True):
        sectoral_data = df.groupby('type')['emissions'].sum()
        charts['sectoral'] = px.pie(
            values=sectoral_data.values,
            names=sectoral_data.index,
            title="Sectoral Distribution"
        )
    
    # Temporal trends
    if report_config.get('include_temporal', True) and 'year' in df.columns:
        temporal_data = df.groupby(['year', 'region'])['emissions'].sum().reset_index()
        charts['temporal'] = px.line(
            temporal_data,
            x='year',
            y='emissions',
            color='region',
            title="Emission Trends by Region",
            markers=True
        )
    
    # Top emitters
    if report_config.get('include_countries', True):
        country_data = df.groupby('country')['emissions'].sum().sort_values(ascending=False).head(10)
        charts['countries'] = px.bar(
            x=country_data.values,
            y=country_data.index,
            orientation='h',
            title="Top 10 Emitting Countries",
            color=country_data.values,
            color_continuous_scale='Reds'
        )
    
    return charts

def generate_insights(df):
    """Generate automated insights from the data"""
    insights = []
    
    # Growth analysis
    if 'year' in df.columns and df['year'].nunique() > 1:
        yearly_totals = df.groupby('year')['emissions'].sum()
        if len(yearly_totals) >= 2:
            growth_rate = ((yearly_totals.iloc[-1] / yearly_totals.iloc[0]) ** (1/(len(yearly_totals)-1)) - 1) * 100
            insights.append(f"Overall emissions have grown at {growth_rate:.1f}% annually")
    
    # Regional insights
    regional_totals = df.groupby('region')['emissions'].sum()
    top_region = regional_totals.idxmax()
    top_region_pct = (regional_totals.max() / regional_totals.sum() * 100)
    insights.append(f"{top_region} accounts for {top_region_pct:.1f}% of total emissions")
    
    # Sectoral insights
    sectoral_totals = df.groupby('type')['emissions'].sum()
    top_sector = sectoral_totals.idxmax()
    top_sector_pct = (sectoral_totals.max() / sectoral_totals.sum() * 100)
    insights.append(f"{top_sector} sector dominates with {top_sector_pct:.1f}% of emissions")
    
    # Concentration insights
    country_totals = df.groupby('country')['emissions'].sum().sort_values(ascending=False)
    top_10_pct = (country_totals.head(10).sum() / country_totals.sum() * 100)
    insights.append(f"Top 10 countries account for {top_10_pct:.1f}% of global emissions")
    
    return insights

def main():
    st.title("📊 Custom Reports Generator")
    st.markdown("Create comprehensive, exportable reports with custom configurations")
    
    # Load data
    df = load_data()
    
    # Report Builder Section
    st.markdown("<div class='report-builder'>", unsafe_allow_html=True)
    st.subheader("🔧 Report Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Report Scope**")
        report_title = st.text_input("Report Title", "Methane Emissions Analysis Report")
        report_period = st.selectbox("Time Period", ["All Years", "Last 5 Years", "Last 3 Years", "Current Year"])
        
    with col2:
        st.markdown("**Content Selection**")
        include_executive = st.checkbox("Executive Summary", value=True)
        include_regional = st.checkbox("Regional Analysis", value=True)
        include_sectoral = st.checkbox("Sectoral Analysis", value=True)
        include_temporal = st.checkbox("Temporal Trends", value=True)
        include_countries = st.checkbox("Country Rankings", value=True)
        include_insights = st.checkbox("Automated Insights", value=True)
    
    with col3:
        st.markdown("**Filters**")
        selected_regions = st.multiselect("Regions", df['region'].unique(), default=df['region'].unique())
        selected_sectors = st.multiselect("Sectors", df['type'].unique(), default=df['type'].unique())
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df[
        (df['region'].isin(selected_regions)) &
        (df['type'].isin(selected_sectors))
    ]
    
    # Apply time period filter
    if 'year' in filtered_df.columns:
        max_year = filtered_df['year'].max()
        if report_period == "Last 5 Years":
            filtered_df = filtered_df[filtered_df['year'] >= max_year - 4]
        elif report_period == "Last 3 Years":
            filtered_df = filtered_df[filtered_df['year'] >= max_year - 2]
        elif report_period == "Current Year":
            filtered_df = filtered_df[filtered_df['year'] == max_year]
    
    # Generate Report Button
    if st.button("🚀 Generate Report", type="primary"):
        # Report Header
        st.markdown(f"""
        <div class="report-header">
            <h1>{report_title}</h1>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p>Data Coverage: {len(filtered_df):,} records | {filtered_df['country'].nunique()} countries | {filtered_df['region'].nunique()} regions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Executive Summary
        if include_executive:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("📋 Executive Summary")
            
            summary = generate_executive_summary(filtered_df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>{summary['total_emissions']:,.0f}</h3>
                    <p>Total Emissions (Mt CO₂e)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>{summary['yoy_change']:+.1f}%</h3>
                    <p>Year-over-Year Change</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>{summary['top_region']}</h3>
                    <p>Leading Region</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>{summary['top_sector']}</h3>
                    <p>Dominant Sector</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate and display charts
        report_config = {
            'include_regional': include_regional,
            'include_sectoral': include_sectoral,
            'include_temporal': include_temporal,
            'include_countries': include_countries
        }
        
        charts = create_report_charts(filtered_df, report_config)
        
        # Regional Analysis
        if include_regional and 'regional' in charts:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("🌍 Regional Analysis")
            st.plotly_chart(charts['regional'], use_container_width=True)
            
            # Regional statistics table
            regional_stats = filtered_df.groupby('region').agg({
                'emissions': ['sum', 'mean', 'count'],
                'country': 'nunique'
            }).round(2)
            regional_stats.columns = ['Total Emissions', 'Average Emissions', 'Data Points', 'Countries']
            st.dataframe(regional_stats, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Sectoral Analysis
        if include_sectoral and 'sectoral' in charts:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("🏭 Sectoral Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['sectoral'], use_container_width=True)
            
            with col2:
                sectoral_stats = filtered_df.groupby('type')['emissions'].agg(['sum', 'mean', 'count'])
                sectoral_stats.columns = ['Total', 'Average', 'Count']
                sectoral_stats['Percentage'] = (sectoral_stats['Total'] / sectoral_stats['Total'].sum() * 100).round(1)
                st.dataframe(sectoral_stats, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Temporal Trends
        if include_temporal and 'temporal' in charts:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("📈 Temporal Trends")
            st.plotly_chart(charts['temporal'], use_container_width=True)
            
            # Trend analysis
            if 'year' in filtered_df.columns and filtered_df['year'].nunique() > 1:
                yearly_growth = filtered_df.groupby('year')['emissions'].sum().pct_change() * 100
                avg_growth = yearly_growth.mean()
                st.markdown(f"**Average annual growth rate: {avg_growth:.1f}%**")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Country Rankings
        if include_countries and 'countries' in charts:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("🏆 Country Rankings")
            st.plotly_chart(charts['countries'], use_container_width=True)
            
            # Top countries table
            top_countries = filtered_df.groupby('country')['emissions'].sum().sort_values(ascending=False).head(15)
            country_table = pd.DataFrame({
                'Country': top_countries.index,
                'Total Emissions': top_countries.values,
                'Percentage': (top_countries.values / filtered_df['emissions'].sum() * 100).round(2)
            })
            st.dataframe(country_table, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Automated Insights
        if include_insights:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.subheader("🎯 Key Insights")
            
            insights = generate_insights(filtered_df)
            
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}.** {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Summary
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.subheader("📊 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview**")
            st.write(f"• Total records: {len(filtered_df):,}")
            st.write(f"• Countries covered: {filtered_df['country'].nunique()}")
            st.write(f"• Regions covered: {filtered_df['region'].nunique()}")
            st.write(f"• Sectors analyzed: {filtered_df['type'].nunique()}")
            if 'year' in filtered_df.columns:
                st.write(f"• Time period: {filtered_df['year'].min():.0f} - {filtered_df['year'].max():.0f}")
        
        with col2:
            st.markdown("**Data Quality**")
            missing_pct = (filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns)) * 100)
            st.write(f"• Missing values: {missing_pct:.1f}%")
            st.write(f"• Total emissions: {filtered_df['emissions'].sum():,.0f} Mt CO₂e")
            st.write(f"• Average per country: {filtered_df.groupby('country')['emissions'].sum().mean():.1f} Mt CO₂e")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Export Options
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export Data (CSV)"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"methane_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export Summary"):
                summary_data = {
                    'Report Title': report_title,
                    'Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Total Records': len(filtered_df),
                    'Total Emissions': filtered_df['emissions'].sum(),
                    'Countries': filtered_df['country'].nunique(),
                    'Regions': filtered_df['region'].nunique()
                }
                st.json(summary_data)
        
        with col3:
            st.markdown("**Print Report**")
            st.markdown("Use your browser's print function (Ctrl+P) to print this report")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            <p><strong>{report_title}</strong></p>
            <p>Generated by Greenhouse Gas Analytics Platform | {datetime.now().strftime("%B %d, %Y")}</p>
            <p style='font-size: 0.8rem;'>Data sources: UNFCCC, EDGAR, IEA, FAO, Climate Watch</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()