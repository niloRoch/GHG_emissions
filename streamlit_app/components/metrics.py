import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """Render a custom metric card with styling"""
    
    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" and str(delta).startswith("+") else "#dc3545" if delta_color == "inverse" else "#17a2b8"
        arrow = "▲" if str(delta).startswith("+") else "▼" if str(delta).startswith("-") else "●"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 0.2rem;">{arrow} {delta}</div>'
    
    help_html = f'<div style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">{help_text}</div>' if help_text else ""
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        margin: 0.3rem 0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.3rem;">{value}</div>
        <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 500;">{title}</div>
        {delta_html}
        {help_html}
    </div>
    """
    
    return card_html

def render_kpi_grid(metrics_data):
    """Render a grid of KPI cards"""
    
    num_metrics = len(metrics_data)
    cols = st.columns(num_metrics)
    
    for i, (col, metric) in enumerate(zip(cols, metrics_data)):
        with col:
            st.markdown(
                render_metric_card(
                    title=metric.get('title', ''),
                    value=metric.get('value', ''),
                    delta=metric.get('delta'),
                    delta_color=metric.get('delta_color', 'normal'),
                    help_text=metric.get('help_text')
                ),
                unsafe_allow_html=True
            )

def render_gauge_chart(value, title, min_val=0, max_val=100, threshold_colors=None):
    """Render a gauge chart for KPIs"""
    
    if threshold_colors is None:
        threshold_colors = ["#FF4444", "#FFAA00", "#00AA00"]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.3], 'color': threshold_colors[0]},
                {'range': [max_val*0.3, max_val*0.7], 'color': threshold_colors[1]},
                {'range': [max_val*0.7, max_val], 'color': threshold_colors[2]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.9
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_sparkline(data, title):
    """Render a sparkline chart for trends"""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        line=dict(color='#667eea', width=2),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def render_status_indicator(status, label):
    """Render a status indicator (good, warning, critical)"""
    
    status_config = {
        'good': {'color': '#28a745', 'icon': '✅', 'bg': '#d4edda'},
        'warning': {'color': '#ffc107', 'icon': '⚠️', 'bg': '#fff3cd'},
        'critical': {'color': '#dc3545', 'icon': '❌', 'bg': '#f8d7da'}
    }
    
    config = status_config.get(status, status_config['warning'])
    
    indicator_html = f"""
    <div style="
        background: {config['bg']};
        border: 1px solid {config['color']};
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    ">
        <span style="font-size: 1.2rem;">{config['icon']}</span>
        <span style="color: {config['color']}; font-weight: 500;">{label}</span>
    </div>
    """
    
    return indicator_html

def render_comparison_bars(data_dict, title="Comparison"):
    """Render horizontal comparison bars"""
    
    max_value = max(data_dict.values()) if data_dict else 1
    
    bars_html = f'<div style="margin: 1rem 0;"><h4>{title}</h4>'
    
    for label, value in data_dict.items():
        percentage = (value / max_value) * 100
        bars_html += f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.2rem;">
                <span>{label}</span>
                <span>{value:,.0f}</span>
            </div>
            <div style="
                width: 100%;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, #667eea, #764ba2);
                    border-radius: 4px;
                "></div>
            </div>
        </div>
        """
    
    bars_html += '</div>'
    return bars_html

def render_trend_indicator(current, previous, label):
    """Render a trend indicator showing change"""
    
    if previous == 0:
        change = 0
        change_pct = 0
    else:
        change = current - previous
        change_pct = (change / previous) * 100
    
    # Determine trend direction and color
    if change > 0:
        trend_icon = "📈"
        trend_color = "#dc3545"  # Red for increasing emissions (bad)
        direction = "increased"
    elif change < 0:
        trend_icon = "📉"
        trend_color = "#28a745"  # Green for decreasing emissions (good)
        direction = "decreased"
    else:
        trend_icon = "➖"
        trend_color = "#6c757d"
        direction = "remained stable"
    
    trend_html = f"""
    <div style="
        background: white;
        border-left: 4px solid {trend_color};
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{trend_icon}</span>
            <strong style="color: {trend_color};">{label}</strong>
        </div>
        <div style="font-size: 0.9rem; color: #666;">
            {direction} by <strong style="color: {trend_color};">{abs(change_pct):.1f}%</strong>
            <br>
            <small>({current:,.1f} vs {previous:,.1f})</small>
        </div>
    </div>
    """
    
    return trend_html

def render_progress_ring(percentage, title, size=120):
    """Render a progress ring/donut chart"""
    
    # Determine color based on percentage
    if percentage >= 80:
        color = "#28a745"
    elif percentage >= 60:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    fig = go.Figure(data=[go.Pie(
        labels=['Progress', 'Remaining'],
        values=[percentage, 100-percentage],
        hole=.7,
        marker_colors=[color, '#e9ecef'],
        textinfo='none',
        hoverinfo='none',
        showlegend=False
    )])
    
    fig.update_layout(
        height=size,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:24px'>{percentage:.1f}%</span>",
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle'
        ),
        annotations=[
            dict(
                x=0.5, y=0.5,
                text=f"{percentage:.1f}%",
                showarrow=False,
                font=dict(size=20, color=color)
            )
        ]
    )
    
    return fig

def create_alert_box(message, alert_type="info"):
    """Create styled alert boxes"""
    
    alert_config = {
        'info': {'color': '#0c5460', 'bg': '#d1ecf1', 'border': '#bee5eb', 'icon': 'ℹ️'},
        'success': {'color': '#155724', 'bg': '#d4edda', 'border': '#c3e6cb', 'icon': '✅'},
        'warning': {'color': '#856404', 'bg': '#fff3cd', 'border': '#ffeaa7', 'icon': '⚠️'},
        'danger': {'color': '#721c24', 'bg': '#f8d7da', 'border': '#f5c6cb', 'icon': '❌'}
    }
    
    config = alert_config.get(alert_type, alert_config['info'])
    
    alert_html = f"""
    <div style="
        color: {config['color']};
        background-color: {config['bg']};
        border: 1px solid {config['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    ">
        <span style="font-size: 1.2rem; flex-shrink: 0;">{config['icon']}</span>
        <div>{message}</div>
    </div>
    """
    
    return alert_html