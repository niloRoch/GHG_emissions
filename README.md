# 🌍 Greenhouse Gas Analytics - Methane Emissions Dashboard

A comprehensive data science project analyzing global methane emissions patterns across different sectors, regions, and time periods.

## 📊 Project Overview

This interactive dashboard provides deep insights into methane emissions worldwide, helping understand:
- Global emission patterns and trends
- Sectoral analysis (Agriculture, Energy, Waste, Other)
- Geographic distribution and hotspots
- Temporal trends and forecasting
- Comparative analysis between regions and countries

## 🏗️ Architecture

```
greenhouse_gas_analytics/
├── 📊 data/
│   ├── raw/                    # Dados brutos
│   ├── processed/              # Dados processados
│   └── exports/                # Dados para exportação
│
├── 🎨 streamlit_app/
│   ├── 🏠 Home.py              # Página principal
│   ├── pages/                  # Páginas multipage
│   ├── components/             # Componentes reutilizáveis
│   └── assets/                 # Estilos e recursos
│
├── 📋 notebooks/               # Jupyter notebooks
└── requirements.txt            # Dependências
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run streamlit_app/Home.py
   ```

3. **Access the dashboard:**
   Open your browser to `http://localhost:8501`

## 📈 Features

### 🏠 Home Dashboard
- Executive summary with key metrics
- Global emission overview
- Top emitters and trends

### 🌍 Global Overview
- World map visualization
- Regional comparisons
- Time series analysis

### 🏭 Sector Analysis
- Detailed breakdown by emission types
- Agriculture vs Energy vs Waste analysis
- Subsector deep dives

### 🗺️ Geographic Insights
- Country-level analysis
- Regional clustering
- Geospatial patterns

### 📈 Trends & Forecasting
- Historical trend analysis
- Predictive modeling
- Scenario planning

### 🔍 Deep Dive Explorer
- Custom filtering and analysis
- Interactive data exploration
- Advanced visualizations

### 📊 Custom Reports
- Exportable reports
- Custom charts generation
- Data download capabilities

## 🎯 Key Insights Generated

- **Top emitting sectors** and their contribution patterns
- **Geographic hotspots** requiring attention
- **Temporal trends** showing improvement or deterioration
- **Sector-specific patterns** (Agriculture vs Energy emissions)
- **Country rankings** and comparative analysis
- **Forecasting models** for future emissions

## 🛠️ Technologies Used

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualizations:** Plotly, Seaborn, Matplotlib, Altair
- **Geospatial:** Folium, Streamlit-Folium
- **Machine Learning:** Scikit-learn, Statsmodels
- **Styling:** Custom CSS, Streamlit Components

## 📊 Data Sources

The dataset includes methane emissions data from various authoritative sources:
- UNFCCC (United Nations Framework Convention on Climate Change)
- EDGAR (Emissions Database for Global Atmospheric Research)
- IEA (International Energy Agency)
- FAO (Food and Agriculture Organization)
- Climate Watch Data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sources: UNFCCC, EDGAR, IEA, FAO, Climate Watch
- Streamlit community for amazing components
- Plotly for interactive visualizations