import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional

# Define target resolution times by issue type (in hours)
# These are example targets that would need to be adjusted based on actual Miami-Dade policies
RESOLUTION_TARGETS = {
    'POTHOLE': 72,  # 3 days
    'STREETLIGHT': 120,  # 5 days
    'TRAFFIC SIGNAL': 24,  # 1 day
    'GRAFFITI': 168,  # 7 days
    'TRASH': 48,  # 2 days
    'WATER LEAK': 24,  # 1 day
    'DRAINAGE': 72,  # 3 days
    'SIDEWALK': 336,  # 14 days
    'TREE': 120,  # 5 days
    'DEFAULT': 120  # Default target: 5 days
}

def calculate_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate resolution time for each service request
    
    Args:
        df: DataFrame with 311 service request data
        
    Returns:
        DataFrame with added resolution time columns
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Identify date columns based on common patterns
    created_date_col = next((col for col in df_copy.columns 
                           if 'creat' in col.lower() and 'date' in col.lower()), None)
    closed_date_col = next((col for col in df_copy.columns 
                          if ('clos' in col.lower() or 'resolv' in col.lower() or 'complet' in col.lower()) 
                          and 'date' in col.lower()), None)
    
    # If we can't find the expected columns, try some common default names
    if created_date_col is None:
        potential_cols = ['CREATED_DATE', 'CREATION_DATE', 'OPEN_DATE', 'CREATEDATE', 'SR_CREATE_DATE', 'CreateDate']
        created_date_col = next((col for col in potential_cols if col in df_copy.columns), None)
    
    if closed_date_col is None:
        potential_cols = ['CLOSED_DATE', 'CLOSURE_DATE', 'RESOLUTION_DATE', 'CLOSEDATE', 'SR_CLOSED_DATE', 'CloseDate']
        closed_date_col = next((col for col in potential_cols if col in df_copy.columns), None)
    
    # Make sure we have both date columns
    if created_date_col is None or closed_date_col is None:
        # If we can't find the date columns, set a dummy resolution time
        df_copy['resolution_hours'] = np.nan
        df_copy['is_resolved'] = False
        return df_copy
    
    # Convert date columns to datetime - making them timezone-naive explicitly
    df_copy[created_date_col] = pd.to_datetime(df_copy[created_date_col], errors='coerce').dt.tz_localize(None)
    df_copy[closed_date_col] = pd.to_datetime(df_copy[closed_date_col], errors='coerce').dt.tz_localize(None)
    
    # Calculate resolution time
    df_copy['resolution_hours'] = (df_copy[closed_date_col] - df_copy[created_date_col]).dt.total_seconds() / 3600
    
    # Not all tickets are resolved
    df_copy['is_resolved'] = ~df_copy[closed_date_col].isna()
    
    # For ongoing tickets, calculate hours since creation
    # Use timezone-naive current date for consistency
    current_date = pd.Timestamp.now().tz_localize(None)
    df_copy['hours_since_creation'] = (current_date - df_copy[created_date_col]).dt.total_seconds() / 3600
    
    # For unresolved tickets, use hours_since_creation as resolution_hours
    df_copy.loc[~df_copy['is_resolved'], 'resolution_hours'] = df_copy.loc[~df_copy['is_resolved'], 'hours_since_creation']
    
    return df_copy

def determine_target_resolution_time(sr_type: str) -> float:
    """
    Determine the target resolution time for a service request type
    
    Args:
        sr_type: Service request type
        
    Returns:
        Target resolution time in hours
    """
    # Convert to uppercase for matching
    sr_type = str(sr_type).upper() if sr_type else ""
    
    # Look for specific issue types in the service request type string
    for issue, target in RESOLUTION_TARGETS.items():
        if issue in sr_type:
            return target
            
    # Default target
    return RESOLUTION_TARGETS['DEFAULT']

def classify_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each request's performance based on resolution time vs target
    
    Args:
        df: DataFrame with 311 service request data including resolution_hours
        
    Returns:
        DataFrame with added performance columns
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Find the service request type column
    sr_type_col = next((col for col in df_copy.columns 
                         if ('type' in col.lower() or 'category' in col.lower() or 'issue' in col.lower())
                        and 'sr' in col.lower()), None)
    
    # If we can't find it, try some common names
    if sr_type_col is None:
        potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
        sr_type_col = next((col for col in potential_cols if col in df_copy.columns), 'SR_TYPE')
    
    # If column still not found, create a dummy one
    if sr_type_col not in df_copy.columns:
        df_copy[sr_type_col] = 'UNKNOWN'
    
    # Get target resolution time for each request
    df_copy['target_hours'] = df_copy[sr_type_col].apply(determine_target_resolution_time)
    
    # Calculate performance ratio (resolution time / target time)
    df_copy['performance_ratio'] = df_copy['resolution_hours'] / df_copy['target_hours']
    
    # Classify performance
    conditions = [
        (df_copy['performance_ratio'] <= 0.5),  # Excellent: half the target time or less
        (df_copy['performance_ratio'] > 0.5) & (df_copy['performance_ratio'] <= 0.9),  # Good: under target
        (df_copy['performance_ratio'] > 0.9) & (df_copy['performance_ratio'] <= 1.1),  # On target
        (df_copy['performance_ratio'] > 1.1) & (df_copy['performance_ratio'] <= 2.0),  # Delayed
        (df_copy['performance_ratio'] > 2.0)  # Severely delayed
    ]
    
    choices = ['Excellent', 'Good', 'On Target', 'Delayed', 'Severely Delayed']
    df_copy['performance_category'] = np.select(conditions, choices, default='Unknown')
    
    # Set performance for unresolved tickets based on time elapsed vs target
    unresolved_conditions = [
        (~df_copy['is_resolved']) & (df_copy['performance_ratio'] <= 0.9),  # Still within target
        (~df_copy['is_resolved']) & (df_copy['performance_ratio'] > 0.9)  # Exceeding target
    ]
    
    unresolved_choices = ['In Progress (Within Target)', 'In Progress (Exceeding Target)']
    mask = np.zeros(len(df_copy), dtype=bool)
    for condition in unresolved_conditions:
        mask = mask | condition
    
    df_copy.loc[mask, 'performance_category'] = np.select(
        unresolved_conditions, unresolved_choices, default=df_copy['performance_category']
    )[mask]
    
    return df_copy

def generate_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Generate performance metrics for service requests
    
    Args:
        df: DataFrame with 311 service request data including performance classifications
        
    Returns:
        Dictionary with various performance metrics
    """
    metrics = {}
    
    # Overall performance metrics
    metrics['total_requests'] = len(df)
    metrics['resolved_requests'] = df['is_resolved'].sum()
    metrics['resolution_rate'] = df['is_resolved'].mean() * 100
    
    # Request statuses
    status_col = next((col for col in df.columns if 'status' in col.lower()), None)
    if status_col and status_col in df.columns:
        metrics['status_counts'] = df[status_col].value_counts().to_dict()
    
    # Performance category breakdowns
    if 'performance_category' in df.columns:
        metrics['performance_categories'] = df['performance_category'].value_counts().to_dict()
        
        # Calculate percentage meeting target for resolved tickets
        resolved_df = df[df['is_resolved']]
        if len(resolved_df) > 0:
            meeting_target = resolved_df['performance_ratio'] <= 1.1  # On target or better
            metrics['percent_meeting_target'] = meeting_target.mean() * 100
        else:
            metrics['percent_meeting_target'] = 0
    
    # Average resolution time
    if 'resolution_hours' in df.columns and df['is_resolved'].sum() > 0:
        metrics['avg_resolution_hours'] = df.loc[df['is_resolved'], 'resolution_hours'].mean()
        metrics['median_resolution_hours'] = df.loc[df['is_resolved'], 'resolution_hours'].median()
    
    # Type-specific metrics
    sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
    if sr_type_col and sr_type_col in df.columns:
        # Get top 10 request types
        top_types = df[sr_type_col].value_counts().head(10).index.tolist()
        
        type_metrics = {}
        for sr_type in top_types:
            type_df = df[df[sr_type_col] == sr_type]
            
            type_metrics[sr_type] = {
                'count': len(type_df),
                'avg_resolution_hours': type_df.loc[type_df['is_resolved'], 'resolution_hours'].mean() 
                    if type_df['is_resolved'].sum() > 0 else np.nan,
                'resolution_rate': type_df['is_resolved'].mean() * 100,
                'target_hours': type_df['target_hours'].iloc[0] if len(type_df) > 0 else np.nan
            }
            
            if 'performance_category' in type_df.columns:
                type_metrics[sr_type]['performance_categories'] = type_df['performance_category'].value_counts().to_dict()
                
                # Meeting target rate for each type
                resolved_type_df = type_df[type_df['is_resolved']]
                if len(resolved_type_df) > 0:
                    meeting_target = resolved_type_df['performance_ratio'] <= 1.1
                    type_metrics[sr_type]['percent_meeting_target'] = meeting_target.mean() * 100
                else:
                    type_metrics[sr_type]['percent_meeting_target'] = 0
        
        metrics['request_types'] = type_metrics
    
    # Geographic metrics
    zipcode_col = next((col for col in df.columns if 'zip' in col.lower()), None)
    if zipcode_col and zipcode_col in df.columns:
        # Get top 10 zipcodes
        top_zipcodes = df[zipcode_col].value_counts().head(10).index.tolist()
        
        zipcode_metrics = {}
        for zipcode in top_zipcodes:
            zipcode_df = df[df[zipcode_col] == zipcode]
            
            zipcode_metrics[zipcode] = {
                'count': len(zipcode_df),
                'avg_resolution_hours': zipcode_df.loc[zipcode_df['is_resolved'], 'resolution_hours'].mean()
                    if zipcode_df['is_resolved'].sum() > 0 else np.nan,
                'resolution_rate': zipcode_df['is_resolved'].mean() * 100
            }
            
            if 'performance_category' in zipcode_df.columns:
                zipcode_metrics[zipcode]['percent_meeting_target'] = (
                    zipcode_df.loc[zipcode_df['is_resolved'], 'performance_ratio'] <= 1.1
                ).mean() * 100 if zipcode_df['is_resolved'].sum() > 0 else 0
        
        metrics['zipcode_metrics'] = zipcode_metrics
    
    # Time trends
    created_date_col = next((col for col in df.columns if 'creat' in col.lower() and 'date' in col.lower()), None)
    if created_date_col and created_date_col in df.columns:
        df[created_date_col] = pd.to_datetime(df[created_date_col], errors='coerce')
        
        # Group by month
        df['month'] = df[created_date_col].dt.to_period('M')
        monthly_counts = df.groupby('month').size()
        
        # Convert period index to strings for serialization
        metrics['monthly_counts'] = {str(k): int(v) for k, v in monthly_counts.items()}
        
        # Performance over time
        if 'performance_ratio' in df.columns and df['is_resolved'].sum() > 0:
            monthly_performance = df[df['is_resolved']].groupby('month')['performance_ratio'].mean()
            metrics['monthly_performance'] = {str(k): float(v) for k, v in monthly_performance.items()}
    
    return metrics

def analyze_service_request_goals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Full pipeline to analyze service request goals
    
    Args:
        df: DataFrame with 311 service request data
        
    Returns:
        Tuple of (DataFrame with added performance analysis columns, metrics dictionary)
    """
    # Calculate resolution times
    df_with_resolution = calculate_resolution_time(df)
    
    # Classify performance
    df_with_performance = classify_performance(df_with_resolution)
    
    # Generate metrics
    metrics = generate_performance_metrics(df_with_performance)
    
    # Return both the DataFrame and metrics as a tuple instead of attaching metrics as an attribute
    return df_with_performance, metrics 