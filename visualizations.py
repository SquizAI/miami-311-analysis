import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def plot_priority_distribution(df):
    """
    Plot the distribution of service request priorities
    
    Args:
        df: DataFrame with 311 data containing PRIORITY column
    """
    if 'PRIORITY' not in df.columns:
        st.warning("Priority data not available in this dataset")
        return
    
    # Get the counts by priority
    priority_counts = df['PRIORITY'].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    
    # Sort in a logical order (HIGH, MEDIUM, NORMAL, LOW)
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    priority_counts['Priority'] = pd.Categorical(
        priority_counts['Priority'], 
        categories=priority_order,
        ordered=True
    )
    priority_counts = priority_counts.sort_values('Priority')
    
    # Create the plot with color coding
    fig = px.bar(
        priority_counts, 
        x='Priority', 
        y='Count',
        color='Priority',
        color_discrete_map={
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'NORMAL': 'blue',
            'LOW': 'green'
        },
        title='Distribution of Service Request Priorities'
    )
    
    fig.update_layout(
        xaxis_title='Priority Level',
        yaxis_title='Number of Requests',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_priority_by_type(df):
    """
    Plot priority distribution by service request type
    
    Args:
        df: DataFrame with 311 data containing PRIORITY and request type columns
    """
    if 'PRIORITY' not in df.columns:
        st.warning("Priority data not available in this dataset")
        return
    
    # Find the service request type column
    sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
    if sr_type_col is None:
        potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
        sr_type_col = next((col for col in potential_cols if col in df.columns), None)
    
    if sr_type_col is None or sr_type_col not in df.columns:
        st.warning("Service request type data not available")
        return
    
    # Get top request types
    top_types = df[sr_type_col].value_counts().head(8).index
    filtered_df = df[df[sr_type_col].isin(top_types)]
    
    # Create a crosstab of type vs priority
    priority_by_type = pd.crosstab(
        filtered_df[sr_type_col], 
        filtered_df['PRIORITY']
    ).reset_index()
    
    # Melt for plotting
    melted_df = pd.melt(
        priority_by_type, 
        id_vars=[sr_type_col], 
        var_name='Priority',
        value_name='Count'
    )
    
    # Sort priorities in logical order
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    melted_df['Priority'] = pd.Categorical(
        melted_df['Priority'], 
        categories=priority_order,
        ordered=True
    )
    
    # Create the plot
    fig = px.bar(
        melted_df,
        x=sr_type_col,
        y='Count',
        color='Priority',
        color_discrete_map={
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'NORMAL': 'blue',
            'LOW': 'green'
        },
        title='Priority Distribution by Service Request Type',
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis_title='Service Request Type',
        yaxis_title='Number of Requests',
        xaxis={'categoryorder': 'total descending'},
        height=500,
        legend_title='Priority Level'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_method_received_distribution(df):
    """
    Plot the distribution of methods by which service requests were received
    
    Args:
        df: DataFrame with 311 data containing METHOD_RECEIVED column
    """
    if 'METHOD_RECEIVED' not in df.columns:
        st.warning("Method received data not available in this dataset")
        return
    
    # Get the counts by method
    method_counts = df['METHOD_RECEIVED'].value_counts().reset_index()
    method_counts.columns = ['Method', 'Count']
    
    # Sort by count
    method_counts = method_counts.sort_values('Count', ascending=False)
    
    # Create the plot with color scheme
    fig = px.bar(
        method_counts, 
        x='Method', 
        y='Count',
        color='Method',
        title='Distribution of Service Request Methods'
    )
    
    fig.update_layout(
        xaxis_title='Method Received',
        yaxis_title='Number of Requests',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_performance_by_method(df):
    """
    Plot performance metrics by method received
    
    Args:
        df: DataFrame with 311 data containing METHOD_RECEIVED and performance metrics
    """
    if 'METHOD_RECEIVED' not in df.columns or 'performance_ratio' not in df.columns:
        st.warning("Method received or performance data not available")
        return
    
    # Calculate average performance ratio by method
    performance_by_method = df.groupby('METHOD_RECEIVED')['performance_ratio'].agg(
        ['mean', 'median', 'count']
    ).reset_index()
    
    # Rename columns for clarity
    performance_by_method.columns = ['Method', 'Mean Performance Ratio', 'Median Performance Ratio', 'Count']
    
    # Sort by count
    performance_by_method = performance_by_method.sort_values('Count', ascending=False)
    
    # Filter to methods with sufficient data
    performance_by_method = performance_by_method[performance_by_method['Count'] > 10]
    
    # Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bars for count
    fig.add_trace(
        go.Bar(
            x=performance_by_method['Method'],
            y=performance_by_method['Count'],
            name='Number of Requests',
            marker_color='lightblue'
        ),
        secondary_y=False
    )
    
    # Add line for mean performance ratio
    fig.add_trace(
        go.Scatter(
            x=performance_by_method['Method'],
            y=performance_by_method['Mean Performance Ratio'],
            name='Mean Performance Ratio',
            mode='lines+markers',
            marker_color='red',
            line=dict(width=3)
        ),
        secondary_y=True
    )
    
    # Add reference line at 1.0 (meeting target)
    fig.add_trace(
        go.Scatter(
            x=performance_by_method['Method'],
            y=[1.0] * len(performance_by_method),
            name='Target',
            mode='lines',
            marker_color='green',
            line=dict(width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Performance by Method Received',
        xaxis_title='Method Received',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Number of Requests", secondary_y=False)
    fig.update_yaxes(title_text="Performance Ratio (Lower is Better)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_priority_performance_heatmap(df):
    """
    Plot a heatmap of performance by priority level
    
    Args:
        df: DataFrame with 311 data containing PRIORITY and performance metrics
    """
    if 'PRIORITY' not in df.columns or 'performance_ratio' not in df.columns:
        st.warning("Priority or performance data not available")
        return
    
    # Calculate average performance and resolution time by priority
    performance_by_priority = df.groupby('PRIORITY').agg(
        avg_performance=('performance_ratio', 'mean'),
        median_performance=('performance_ratio', 'median'),
        avg_resolution_hours=('resolution_hours', 'mean'),
        median_resolution_hours=('resolution_hours', 'median'),
        count=('PRIORITY', 'count'),
        target_met_pct=('performance_ratio', lambda x: (x <= 1.1).mean() * 100)
    ).reset_index()
    
    # Sort in logical order
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    performance_by_priority['PRIORITY'] = pd.Categorical(
        performance_by_priority['PRIORITY'], 
        categories=priority_order,
        ordered=True
    )
    performance_by_priority = performance_by_priority.sort_values('PRIORITY')
    
    # Create a heatmap-style table
    performance_by_priority['Performance Score'] = performance_by_priority['target_met_pct']
    
    # Prepare the figure
    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=[
                    'Priority Level', 
                    'Count', 
                    'Avg Resolution (hrs)', 
                    'Target Met %', 
                    'Performance Score'
                ],
                fill_color='royalblue',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    performance_by_priority['PRIORITY'],
                    performance_by_priority['count'],
                    performance_by_priority['avg_resolution_hours'].round(1),
                    performance_by_priority['target_met_pct'].round(1),
                    performance_by_priority['Performance Score'].round(1)
                ],
                fill_color=[
                    'lightgrey',
                    'lightgrey',
                    'lightgrey',
                    'lightgrey',
                    [
                        # Color scale from red to green based on performance score
                        f'rgba({max(0, min(255, int(255 * (1 - value/100))))}, {max(0, min(255, int(255 * (value/100))))}, 0, 0.7)'
                        for value in performance_by_priority['Performance Score']
                    ]
                ],
                align='left',
                height=30
            )
        )
    ])
    
    fig.update_layout(
        title='Performance Metrics by Priority Level',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_map_by_priority(df):
    """
    Plot service requests on a map, colored by priority
    
    Args:
        df: DataFrame with 311 data containing coordinates and PRIORITY
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("Coordinate data not available for mapping")
        return
    
    # Filter to records with valid coordinates
    if 'has_valid_coords' in df.columns:
        map_data = df[df['has_valid_coords']]
    else:
        # Filter out likely invalid coordinates
        map_data = df.dropna(subset=['latitude', 'longitude'])
        map_data = map_data[
            (map_data['latitude'] > 25.0) & (map_data['latitude'] < 26.5) &
            (map_data['longitude'] > -81.0) & (map_data['longitude'] < -80.0)
        ]
    
    if len(map_data) == 0:
        st.warning("No valid coordinate data available for mapping")
        return
    
    # Limit to a random sample of 5000 points if too many (for performance)
    if len(map_data) > 5000:
        map_data = map_data.sample(5000, random_state=42)
    
    # Set priority for coloring if available
    if 'PRIORITY' in map_data.columns:
        color_col = 'PRIORITY'
        color_map = {
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'NORMAL': 'blue',
            'LOW': 'green'
        }
        
        # Ensure priority is in the right order for the legend
        priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
        map_data['PRIORITY_ORD'] = pd.Categorical(
            map_data['PRIORITY'], 
            categories=priority_order,
            ordered=True
        )
        map_data = map_data.sort_values('PRIORITY_ORD')
        
        title = 'Service Requests by Priority'
    else:
        # Fallback to service request type if priority not available
        sr_type_col = next((col for col in map_data.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
        if sr_type_col:
            color_col = sr_type_col
        else:
            color_col = None
        
        color_map = None
        title = 'Service Request Locations'
    
    # Determine hover data
    hover_data = ['PRIORITY'] if 'PRIORITY' in map_data.columns else None
    
    # Add service request type to hover if available
    sr_type_col = next((col for col in map_data.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
    if sr_type_col:
        if hover_data:
            hover_data.append(sr_type_col)
        else:
            hover_data = [sr_type_col]
    
    # Create the map using scatter_map instead of deprecated scatter_mapbox
    fig = px.scatter_map(
        map_data,
        lat='latitude',
        lon='longitude',
        color=color_col,
        color_discrete_map=color_map,
        hover_name=sr_type_col if sr_type_col else None,
        hover_data=hover_data,
        zoom=10,
        map_style='open-street-map',
        title=title
    )
    
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_performance_by_priority_type(df):
    """
    Create a multi-faceted visualization showing performance by priority and request type
    
    Args:
        df: DataFrame with 311 data containing PRIORITY, request type, and performance metrics
    """
    if 'PRIORITY' not in df.columns or 'performance_ratio' not in df.columns:
        st.warning("Priority or performance data not available")
        return
    
    # Find the service request type column
    sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
    if sr_type_col is None:
        potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
        sr_type_col = next((col for col in potential_cols if col in df.columns), None)
    
    if sr_type_col is None or sr_type_col not in df.columns:
        st.warning("Service request type data not available")
        return
        
    # Get top request types
    top_types = df[sr_type_col].value_counts().head(5).index
    filtered_df = df[df[sr_type_col].isin(top_types)]
    
    # Sort priorities in logical order
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    filtered_df['PRIORITY'] = pd.Categorical(
        filtered_df['PRIORITY'], 
        categories=priority_order,
        ordered=True
    )
    
    # Aggregate data
    performance_data = filtered_df.groupby([sr_type_col, 'PRIORITY']).agg(
        avg_perf=('performance_ratio', 'mean'),
        median_perf=('performance_ratio', 'median'),
        target_met=('performance_ratio', lambda x: (x <= 1.1).mean() * 100),
        count=('performance_ratio', 'count')
    ).reset_index()
    
    # Create a bubble chart
    fig = px.scatter(
        performance_data,
        x='PRIORITY',
        y=sr_type_col,
        size='count',
        color='target_met',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        hover_data=['avg_perf', 'median_perf', 'target_met', 'count'],
        title='Performance by Priority and Request Type',
        labels={
            'target_met': 'Target Met %',
            'avg_perf': 'Avg Perf Ratio',
            'median_perf': 'Median Perf Ratio',
            'count': 'Number of Requests'
        }
    )
    
    fig.update_layout(
        xaxis_title='Priority Level',
        yaxis_title='Service Request Type',
        height=500,
        coloraxis_colorbar=dict(title='Target Met %')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_method_priority_analysis(df):
    """
    Plot analysis of priority by method received
    
    Args:
        df: DataFrame with 311 data containing METHOD_RECEIVED and PRIORITY
    """
    if 'METHOD_RECEIVED' not in df.columns or 'PRIORITY' not in df.columns:
        st.warning("Method received or priority data not available")
        return
    
    # Create a crosstab
    method_priority = pd.crosstab(
        df['METHOD_RECEIVED'],
        df['PRIORITY'],
        normalize='index'
    ) * 100
    
    # Ensure all priority columns exist
    for priority in ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']:
        if priority not in method_priority.columns:
            method_priority[priority] = 0
    
    # Sort priorities in logical order
    method_priority = method_priority[['HIGH', 'MEDIUM', 'NORMAL', 'LOW']]
    
    # Sort methods by HIGH priority percentage
    method_priority = method_priority.sort_values('HIGH', ascending=False)
    
    # Convert to long form for plotting
    method_priority_long = method_priority.reset_index().melt(
        id_vars=['METHOD_RECEIVED'],
        var_name='PRIORITY',
        value_name='Percentage'
    )
    
    # Sort priorities in logical order
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    method_priority_long['PRIORITY'] = pd.Categorical(
        method_priority_long['PRIORITY'], 
        categories=priority_order,
        ordered=True
    )
    
    # Create the plot
    fig = px.bar(
        method_priority_long,
        x='METHOD_RECEIVED',
        y='Percentage',
        color='PRIORITY',
        color_discrete_map={
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'NORMAL': 'blue',
            'LOW': 'green'
        },
        title='Priority Distribution by Method Received (%)',
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis_title='Method Received',
        yaxis_title='Percentage of Requests',
        height=500,
        legend_title='Priority Level'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Now create a plot showing goal achievement by method and priority
    if 'performance_ratio' in df.columns and 'is_resolved' in df.columns:
        # Filter to resolved requests
        resolved_df = df[df['is_resolved']]
        
        # Calculate percentage meeting target by method and priority
        meeting_target = resolved_df.groupby(['METHOD_RECEIVED', 'PRIORITY']).apply(
            lambda x: (x['performance_ratio'] <= 1.1).mean() * 100
        ).reset_index()
        meeting_target.columns = ['METHOD_RECEIVED', 'PRIORITY', 'Percent_Meeting_Target']
        
        # Sort priorities in logical order
        meeting_target['PRIORITY'] = pd.Categorical(
            meeting_target['PRIORITY'], 
            categories=priority_order,
            ordered=True
        )
        
        # Create the plot
        fig = px.bar(
            meeting_target,
            x='METHOD_RECEIVED',
            y='Percent_Meeting_Target',
            color='PRIORITY',
            color_discrete_map={
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'NORMAL': 'blue',
                'LOW': 'green'
            },
            title='Percentage Meeting Target by Method and Priority',
            barmode='group'
        )
        
        # Add a reference line at 80%
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(meeting_target['METHOD_RECEIVED'].unique()) - 0.5,
            y0=80,
            y1=80,
            line=dict(color="black", width=2, dash="dash")
        )
        
        fig.update_layout(
            xaxis_title='Method Received',
            yaxis_title='Percentage Meeting Target',
            height=500,
            legend_title='Priority Level'
        )
        
        st.plotly_chart(fig, use_container_width=True) 