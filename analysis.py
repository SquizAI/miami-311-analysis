import pandas as pd
import streamlit as st

# Import our modules
import performance_analysis as pa
import ai_insights

def analyze_311_data(df, openai_api_key=None):
    """
    Comprehensive analysis of 311 service request data
    
    Args:
        df: DataFrame with 311 data
        openai_api_key: OpenAI API key for AI-powered insights
        
    Returns:
        DataFrame with added analysis columns, metrics dictionary
    """
    # Apply performance analysis
    df_analyzed, metrics = pa.analyze_service_request_goals(df)
    
    return df_analyzed, metrics

def generate_ai_insights(df, metrics, openai_api_key, request_type=None, custom_question=None, analysis_type=None):
    """
    Generate AI-powered insights on the 311 data
    
    Args:
        df: DataFrame with 311 data
        metrics: Dictionary of performance metrics
        openai_api_key: OpenAI API key
        request_type: Optional specific request type to analyze
        custom_question: Optional custom question to analyze
        analysis_type: Optional specific analysis type to perform
        
    Returns:
        Dictionary of insights
    """
    if not openai_api_key:
        return {"error": "OpenAI API key is required for AI insights"}
    
    # Initialize the insight analyzer
    analyzer = ai_insights.InsightAnalyzer(openai_api_key)
    
    # Container for all insights
    insights = {}
    
    # Handle custom question if provided
    if custom_question:
        insights['custom_analysis'] = analyzer.analyze_custom_question(df, custom_question, metrics)
        return insights
    
    # If a specific analysis type is requested
    if analysis_type:
        if analysis_type == "performance":
            insights['performance_overview'] = analyzer.generate_performance_insights(metrics)
        elif analysis_type == "geographic":
            insights['geographic_analysis'] = analyzer.analyze_geographic_patterns(df)
        elif analysis_type == "seasonal":
            insights['seasonal_analysis'] = analyzer.perform_seasonal_analysis(df)
        elif analysis_type == "improvement":
            insights['improvement_plan'] = analyzer.generate_improvement_plan(metrics)
        elif analysis_type.startswith("comparative_"):
            comparison_type = analysis_type.split("_")[1]
            insights['comparative_analysis'] = analyzer.comparative_analysis(df, comparison_type)
        return insights
    
    # If a specific request type is provided, analyze it
    if request_type:
        insights['request_type_analysis'] = analyzer.analyze_request_type(df, request_type)
        return insights
    
    # Default behavior - generate general insights
    insights['performance_overview'] = analyzer.generate_performance_insights(metrics)
    insights['geographic_analysis'] = analyzer.analyze_geographic_patterns(df)
    insights['seasonal_analysis'] = analyzer.perform_seasonal_analysis(df)
    insights['improvement_plan'] = analyzer.generate_improvement_plan(metrics)
    
    return insights

def analyze_priority_goals(df):
    """
    Analyze whether service requests are meeting goals by priority level
    
    Args:
        df: DataFrame with 311 data including PRIORITY and performance_ratio
        
    Returns:
        DataFrame with goal achievement metrics by priority
    """
    if 'PRIORITY' not in df.columns or 'performance_ratio' not in df.columns:
        st.warning("Priority or performance data not available")
        return None
    
    # Calculate goal achievement by priority
    priority_goals = df.groupby('PRIORITY').agg(
        total_requests=('PRIORITY', 'count'),
        avg_resolution_hours=('resolution_hours', 'mean'),
        median_resolution_hours=('resolution_hours', 'median'),
        target_met_count=('performance_ratio', lambda x: (x <= 1.1).sum()),
        target_met_pct=('performance_ratio', lambda x: (x <= 1.1).mean() * 100),
        avg_performance_ratio=('performance_ratio', 'mean')
    ).reset_index()
    
    # Calculate goal status
    priority_goals['goal_status'] = priority_goals['target_met_pct'].apply(
        lambda x: 'Meeting Goal' if x >= 80 else 'Not Meeting Goal'
    )
    
    # Sort in logical order
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    priority_goals['PRIORITY'] = pd.Categorical(
        priority_goals['PRIORITY'], 
        categories=priority_order,
        ordered=True
    )
    priority_goals = priority_goals.sort_values('PRIORITY')
    
    return priority_goals

def analyze_method_received_goals(df):
    """
    Analyze whether service requests are meeting goals by method received
    
    Args:
        df: DataFrame with 311 data including METHOD_RECEIVED and performance_ratio
        
    Returns:
        DataFrame with goal achievement metrics by method
    """
    if 'METHOD_RECEIVED' not in df.columns or 'performance_ratio' not in df.columns:
        st.warning("Method received or performance data not available")
        return None
    
    # Calculate goal achievement by method
    method_goals = df.groupby('METHOD_RECEIVED').agg(
        total_requests=('METHOD_RECEIVED', 'count'),
        avg_resolution_hours=('resolution_hours', 'mean'),
        median_resolution_hours=('resolution_hours', 'median'),
        target_met_count=('performance_ratio', lambda x: (x <= 1.1).sum()),
        target_met_pct=('performance_ratio', lambda x: (x <= 1.1).mean() * 100),
        avg_performance_ratio=('performance_ratio', 'mean')
    ).reset_index()
    
    # Calculate goal status
    method_goals['goal_status'] = method_goals['target_met_pct'].apply(
        lambda x: 'Meeting Goal' if x >= 80 else 'Not Meeting Goal'
    )
    
    # Sort by count
    method_goals = method_goals.sort_values('total_requests', ascending=False)
    
    return method_goals

def analyze_priorities_by_method(df):
    """
    Analyze priority distribution and performance by method received
    
    Args:
        df: DataFrame with 311 data including METHOD_RECEIVED, PRIORITY, and performance metrics
        
    Returns:
        DataFrame with priority metrics by method received
    """
    if 'METHOD_RECEIVED' not in df.columns or 'PRIORITY' not in df.columns:
        st.warning("Method received or priority data not available")
        return None
    
    # Calculate counts by method and priority
    priority_counts = df.groupby(['METHOD_RECEIVED', 'PRIORITY']).size().reset_index(name='count')
    
    # Calculate percentages within each method
    method_totals = priority_counts.groupby('METHOD_RECEIVED')['count'].sum().reset_index()
    priority_counts = priority_counts.merge(method_totals, on='METHOD_RECEIVED', suffixes=('', '_total'))
    priority_counts['percentage'] = priority_counts['count'] / priority_counts['count_total'] * 100
    
    # Add performance metrics if available
    if 'performance_ratio' in df.columns:
        # Calculate average performance ratio and target met percentage
        performance_metrics = df.groupby(['METHOD_RECEIVED', 'PRIORITY']).agg(
            avg_performance_ratio=('performance_ratio', 'mean'),
            target_met_pct=('performance_ratio', lambda x: (x <= 1.1).mean() * 100)
        ).reset_index()
        
        # Merge with priority counts
        priority_counts = priority_counts.merge(performance_metrics, on=['METHOD_RECEIVED', 'PRIORITY'])
    
    # Sort by method and priority
    priority_order = ['HIGH', 'MEDIUM', 'NORMAL', 'LOW']
    priority_counts['PRIORITY'] = pd.Categorical(
        priority_counts['PRIORITY'], 
        categories=priority_order,
        ordered=True
    )
    priority_counts = priority_counts.sort_values(['METHOD_RECEIVED', 'PRIORITY'])
    
    return priority_counts

def analyze_performance_over_time(df):
    """
    Analyze performance trends over time
    
    Args:
        df: DataFrame with 311 data including date columns and performance metrics
        
    Returns:
        DataFrame with performance metrics aggregated by month
    """
    # Find created date column
    created_date_col = next((col for col in df.columns if 'creat' in col.lower() and 'date' in col.lower()), None)
    if created_date_col is None:
        potential_cols = ['CREATED_DATE', 'CREATION_DATE', 'OPEN_DATE', 'CREATEDATE', 'SR_CREATE_DATE']
        created_date_col = next((col for col in potential_cols if col in df.columns), None)
    
    if created_date_col is None or created_date_col not in df.columns:
        st.warning("Creation date data not available")
        return None
    
    # Ensure date column is datetime and timezone-naive for consistency
    df_time = df.copy()
    df_time[created_date_col] = pd.to_datetime(df_time[created_date_col], errors='coerce').dt.tz_localize(None)
    
    # Create month column
    df_time['month'] = df_time[created_date_col].dt.to_period('M')
    
    # Calculate metrics by month
    monthly_metrics = df_time.groupby('month').agg(
        request_count=('month', 'count'),
        avg_resolution_hours=('resolution_hours', 'mean'),
        median_resolution_hours=('resolution_hours', 'median'),
        target_met_pct=('performance_ratio', lambda x: (x <= 1.1).mean() * 100),
        high_priority_pct=('PRIORITY', lambda x: (x == 'HIGH').mean() * 100) if 'PRIORITY' in df_time.columns else None
    ).reset_index()
    
    # Convert period to datetime for easier plotting
    monthly_metrics['month_date'] = monthly_metrics['month'].dt.to_timestamp()
    
    return monthly_metrics 