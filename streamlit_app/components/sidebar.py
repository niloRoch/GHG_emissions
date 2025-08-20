import streamlit as st
import pandas as pd

def render_sidebar(df=None):
    """Render a consistent sidebar across all pages"""
    
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-bottom: 1rem;'>
        <h2>🌍 GHG Analytics</h2>
        <p>Methane Emissions Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation help
    st.sidebar.markdown("### 📖 Page Guide")
    st.sidebar.markdown("""
    - **🏠 Home**: Overview & key metrics
    - **🌍 Global**: World maps & regional analysis
    - **🏭 Sector**: Industry breakdowns
    - **🗺️ Geographic**: Country-level insights
    - **📈 Trends**: Forecasting & scenarios
    - **🔍 Explorer**: Advanced analytics
    - **📊 Reports**: Custom report generation
    """)
    
    # Quick stats if data is provided
    if df is not None:
        st.sidebar.markdown("### 📊 Quick Stats")
        
        total_emissions = df['emissions'].sum() if 'emissions' in df.columns else 0
        total_countries = df['country'].nunique() if 'country' in df.columns else 0
        total_regions = df['region'].nunique() if 'region' in df.columns else 0
        
        st.sidebar.metric("Total Emissions", f"{total_emissions:,.0f} Mt", help="Total methane emissions in Mt CO₂e")
        st.sidebar.metric("Countries", total_countries, help="Number of countries in dataset")
        st.sidebar.metric("Regions", total_regions, help="Number of regions covered")
        
        if 'year' in df.columns and df['year'].notna().any():
            year_range = f"{df['year'].min():.0f}-{df['year'].max():.0f}"
            st.sidebar.metric("Time Span", year_range, help="Years covered in dataset")
    
    # Data sources
    st.sidebar.markdown("### 📚 Data Sources")
    st.sidebar.markdown("""
    <div style='font-size: 0.8rem; color: #666;'>
    • UNFCCC (Climate Convention)<br>
    • EDGAR (EU Emissions Database)<br>
    • IEA (Energy Agency)<br>
    • FAO (Food & Agriculture)<br>
    • Climate Watch Data
    </div>
    """, unsafe_allow_html=True)
    
    # Contact info
    st.sidebar.markdown("### 📞 Support")
    st.sidebar.markdown("""
    <div style='font-size: 0.8rem; color: #666;'>
    For technical support or data inquiries,<br>
    contact the development team.
    </div>
    """, unsafe_allow_html=True)
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.7rem; color: #888;'>
    Dashboard v1.0<br>
    Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)