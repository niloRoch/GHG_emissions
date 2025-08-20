import streamlit as st
import pandas as pd
import numpy as np

def create_multi_filter_panel(df, filter_configs):
    """Create a comprehensive multi-filter panel"""
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
        <h4 style='margin: 0; color: #333;'>🎛️ Advanced Filters</h4>
    </div>
    """, unsafe_allow_html=True)
    
    filters = {}
    
    # Create columns for filter layout
    num_filters = len(filter_configs)
    cols = st.columns(min(num_filters, 4))  # Max 4 columns
    
    for i, (filter_name, config) in enumerate(filter_configs.items()):
        col_index = i % 4
        
        with cols[col_index]:
            if config['type'] == 'multiselect':
                filters[filter_name] = st.multiselect(
                    config['label'],
                    options=config['options'],
                    default=config.get('default', config['options'][:5] if len(config['options']) > 5 else config['options']),
                    key=f"filter_{filter_name}"
                )
            
            elif config['type'] == 'selectbox':
                filters[filter_name] = st.selectbox(
                    config['label'],
                    options=config['options'],
                    index=config.get('default_index', 0),
                    key=f"filter_{filter_name}"
                )
            
            elif config['type'] == 'slider':
                min_val, max_val = config['range']
                filters[filter_name] = st.slider(
                    config['label'],
                    min_value=min_val,
                    max_value=max_val,
                    value=config.get('default', (min_val, max_val)),
                    key=f"filter_{filter_name}"
                )
            
            elif config['type'] == 'date_range':
                filters[filter_name] = st.date_input(
                    config['label'],
                    value=config.get('default'),
                    key=f"filter_{filter_name}"
                )
    
    return filters

def apply_filters(df, filters):
    """Apply multiple filters to dataframe"""
    
    filtered_df = df.copy()
    
    for filter_name, filter_value in filters.items():
        if isinstance(filter_value, list) and len(filter_value) > 0:
            # Multiselect filter
            filtered_df = filtered_df[filtered_df[filter_name].isin(filter_value)]
        
        elif isinstance(filter_value, tuple) and len(filter_value) == 2:
            # Range filter (slider)
            filtered_df = filtered_df[
                (filtered_df[filter_name] >= filter_value[0]) & 
                (filtered_df[filter_name] <= filter_value[1])
            ]
        
        elif filter_value is not None and filter_value != 'All':
            # Single select filter
            filtered_df = filtered_df[filtered_df[filter_name] == filter_value]
    
    return filtered_df

def create_smart_filters(df, primary_columns):
    """Create intelligent filters based on data types and distributions"""
    
    filter_configs = {}
    
    for col in primary_columns:
        if col in df.columns:
            # Determine filter type based on data type and cardinality
            unique_values = df[col].nunique()
            total_values = len(df)
            
            if df[col].dtype == 'object':
                if unique_values <= 50:  # Categorical with reasonable options
                    filter_configs[col] = {
                        'type': 'multiselect',
                        'label': col.replace('_', ' ').title(),
                        'options': sorted(df[col].unique())
                    }
                else:  # Too many categories - use selectbox with top values
                    top_values = df[col].value_counts().head(20).index.tolist()
                    filter_configs[col] = {
                        'type': 'selectbox',
                        'label': f"Top {col.replace('_', ' ').title()}",
                        'options': ['All'] + top_values
                    }
            
            elif df[col].dtype in ['int64', 'float64']:
                min_val, max_val = df[col].min(), df[col].max()
                if unique_values <= 20:  # Few unique values - use multiselect
                    filter_configs[col] = {
                        'type': 'multiselect',
                        'label': col.replace('_', ' ').title(),
                        'options': sorted(df[col].unique())
                    }
                else:  # Continuous - use range slider
                    filter_configs[col] = {
                        'type': 'slider',
                        'label': f"{col.replace('_', ' ').title()} Range",
                        'range': (float(min_val), float(max_val)),
                        'default': (float(min_val), float(max_val))
                    }
    
    return filter_configs

def render_filter_summary(original_df, filtered_df):
    """Render a summary of applied filters"""
    
    reduction_pct = (1 - len(filtered_df) / len(original_df)) * 100
    
    summary_html = f"""
    <div style='
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    '>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <strong>📊 Filter Results</strong><br>
                <small style='color: #666;'>
                    Showing {len(filtered_df):,} of {len(original_df):,} records 
                    ({reduction_pct:.1f}% filtered out)
                </small>
            </div>
            <div style='text-align: right;'>
                <div style='
                    background: {"#28a745" if reduction_pct > 0 else "#6c757d"};
                    color: white;
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: bold;
                '>
                    {"FILTERED" if reduction_pct > 0 else "NO FILTER"}
                </div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(summary_html, unsafe_allow_html=True)

def create_quick_filters(df):
    """Create common quick filter buttons"""
    
    st.markdown("### ⚡ Quick Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_filters = {}
    
    with col1:
        if st.button("🌍 All Regions", key="quick_all_regions"):
            quick_filters['region'] = df['region'].unique().tolist()
    
    with col2:
        if st.button("🏭 Energy Only", key="quick_energy"):
            quick_filters['type'] = ['Energy']
    
    with col3:
        if st.button("🌾 Agriculture Only", key="quick_agriculture"):
            quick_filters['type'] = ['Agriculture']
    
    with col4:
        if st.button("🗑️ Waste Only", key="quick_waste"):
            quick_filters['type'] = ['Waste']
    
    return quick_filters

def create_advanced_text_filter(df, text_columns):
    """Create text-based search filters"""
    
    st.markdown("### 🔍 Text Search")
    
    search_term = st.text_input(
        "Search in data",
        placeholder="Enter search term...",
        help="Search across text columns"
    )
    
    if search_term:
        mask = pd.Series([False] * len(df))
        
        for col in text_columns:
            if col in df.columns:
                mask |= df[col].str.contains(search_term, case=False, na=False)
        
        return df[mask]
    
    return df

def create_comparison_filter(df, comparison_column):
    """Create filters for comparison analysis"""
    
    st.markdown(f"### ⚖️ Compare {comparison_column.title()}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        baseline = st.selectbox(
            "Baseline (Compare From)",
            options=df[comparison_column].unique(),
            key="comparison_baseline"
        )
    
    with col2:
        target = st.selectbox(
            "Target (Compare To)",
            options=df[comparison_column].unique(),
            key="comparison_target"
        )
    
    if baseline != target:
        comparison_data = df[df[comparison_column].isin([baseline, target])]
        
        # Calculate comparison metrics
        baseline_total = df[df[comparison_column] == baseline]['emissions'].sum()
        target_total = df[df[comparison_column] == target]['emissions'].sum()
        
        difference = target_total - baseline_total
        pct_change = (difference / baseline_total * 100) if baseline_total > 0 else 0
        
        # Display comparison summary
        comparison_html = f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        '>
            <h4 style='margin: 0 0 0.5rem 0;'>Comparison Results</h4>
            <div style='display: flex; justify-content: space-between;'>
                <div>
                    <strong>{baseline}:</strong> {baseline_total:,.0f} Mt<br>
                    <strong>{target}:</strong> {target_total:,.0f} Mt
                </div>
                <div style='text-align: right;'>
                    <strong>Difference:</strong> {difference:+,.0f} Mt<br>
                    <strong>Change:</strong> {pct_change:+.1f}%
                </div>
            </div>
        </div>
        """
        
        st.markdown(comparison_html, unsafe_allow_html=True)
        
        return comparison_data, {'baseline': baseline, 'target': target}
    
    return df, None

def save_filter_preset(filters, preset_name):
    """Save current filter configuration as preset"""
    
    if 'filter_presets' not in st.session_state:
        st.session_state.filter_presets = {}
    
    st.session_state.filter_presets[preset_name] = filters
    st.success(f"Filter preset '{preset_name}' saved!")

def load_filter_preset(preset_name):
    """Load a saved filter preset"""
    
    if 'filter_presets' in st.session_state and preset_name in st.session_state.filter_presets:
        return st.session_state.filter_presets[preset_name]
    
    return {}

def render_filter_presets():
    """Render filter preset management interface"""
    
    st.markdown("### 💾 Filter Presets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preset_name = st.text_input("Preset Name", placeholder="My Filter")
    
    with col2:
        if st.button("💾 Save Preset") and preset_name:
            # This would save current filter state - implementation depends on context
            st.info("Preset save functionality would be implemented here")
    
    with col3:
        if 'filter_presets' in st.session_state:
            presets = list(st.session_state.filter_presets.keys())
            if presets:
                selected_preset = st.selectbox("Load Preset", options=presets)
                if st.button("📂 Load"):
                    return load_filter_preset(selected_preset)
    
    return {}