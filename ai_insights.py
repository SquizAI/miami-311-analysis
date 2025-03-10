import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional, Any
from openai import OpenAI

# Add custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

class InsightAnalyzer:
    """Class to generate AI-powered insights from 311 data"""
    
    def __init__(self, api_key: str):
        """
        Initialize the InsightAnalyzer with an OpenAI API key
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
    
    def generate_performance_insights(self, metrics: Dict, max_tokens: int = 1000) -> str:
        """
        Generate insights on performance metrics using OpenAI
        
        Args:
            metrics: Dictionary of performance metrics
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with insights on performance
        """
        # Create a prompt with the metrics - use custom encoder for NumPy types
        metrics_json = json.dumps(metrics, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are performance metrics for Miami-Dade County 311 service requests.
        
        {metrics_json}
        
        Analyze these metrics and provide insights on:
        1. Overall performance in meeting resolution time targets
        2. Which types of requests are performing well vs. poorly
        3. Geographic patterns in resolution times
        4. Trends over time
        5. Prioritized recommendations for improvement
        
        Focus on practical, data-driven insights that would help service managers improve performance.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst and public service expert specializing in performance management for 311 service requests."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating performance insights: {str(e)}"
    
    def analyze_request_type(self, df: pd.DataFrame, request_type: str, max_tokens: int = 1000) -> str:
        """
        Generate detailed analysis of a specific request type
        
        Args:
            df: DataFrame with 311 service request data
            request_type: The specific request type to analyze
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with detailed analysis of the request type
        """
        # Find the SR_TYPE column
        sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
        if sr_type_col is None:
            potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
            sr_type_col = next((col for col in potential_cols if col in df.columns), None)
        
        if sr_type_col is None or sr_type_col not in df.columns:
            return "Could not find service request type column in the data."
        
        # Filter for the specific request type
        request_df = df[df[sr_type_col] == request_type]
        
        if len(request_df) == 0:
            return f"No data found for request type: {request_type}"
        
        # Collect stats for the request type
        type_stats = {
            'request_type': request_type,
            'count': len(request_df),
            'percent_of_total': len(request_df) / len(df) * 100
        }
        
        # Resolution stats if available
        if 'is_resolved' in request_df.columns:
            type_stats['resolved_count'] = request_df['is_resolved'].sum()
            type_stats['resolution_rate'] = request_df['is_resolved'].mean() * 100
        
        if 'resolution_hours' in request_df.columns and request_df['is_resolved'].sum() > 0:
            type_stats['avg_resolution_hours'] = request_df.loc[request_df['is_resolved'], 'resolution_hours'].mean()
            type_stats['median_resolution_hours'] = request_df.loc[request_df['is_resolved'], 'resolution_hours'].median()
        
        if 'target_hours' in request_df.columns:
            type_stats['target_hours'] = request_df['target_hours'].iloc[0]
        
        if 'performance_category' in request_df.columns:
            type_stats['performance_categories'] = request_df['performance_category'].value_counts().to_dict()
            
            # Calculate percentage meeting target
            if 'performance_ratio' in request_df.columns and request_df['is_resolved'].sum() > 0:
                meeting_target = request_df.loc[request_df['is_resolved'], 'performance_ratio'] <= 1.1
                type_stats['percent_meeting_target'] = meeting_target.mean() * 100
        
        # Geographic distribution
        zipcode_col = next((col for col in request_df.columns if 'zip' in col.lower()), None)
        if zipcode_col and zipcode_col in request_df.columns:
            type_stats['top_zipcodes'] = request_df[zipcode_col].value_counts().head(5).to_dict()
        
        # Time trends
        created_date_col = next((col for col in request_df.columns if 'creat' in col.lower() and 'date' in col.lower()), None)
        if created_date_col and created_date_col in request_df.columns:
            request_df[created_date_col] = pd.to_datetime(request_df[created_date_col], errors='coerce')
            request_df['month'] = request_df[created_date_col].dt.to_period('M')
            monthly_counts = request_df.groupby('month').size()
            type_stats['monthly_counts'] = {str(k): int(v) for k, v in monthly_counts.items()}
        
        # Create the prompt - use custom encoder for NumPy types
        stats_json = json.dumps(type_stats, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are statistics for the "{request_type}" service request type in Miami-Dade County's 311 system.
        
        {stats_json}
        
        Analyze these metrics and provide insights on:
        1. How well this service request type is being handled compared to targets
        2. Geographic patterns in where these requests originate
        3. Trends over time (increasing, decreasing, seasonal)
        4. Root causes for any performance issues
        5. Specific recommendations to improve resolution times and customer satisfaction
        
        Focus on actionable insights that would help service managers improve performance for this specific type of request.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst and public service expert specializing in 311 service request performance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing request type: {str(e)}"
    
    def analyze_geographic_patterns(self, df: pd.DataFrame, max_tokens: int = 1000) -> str:
        """
        Generate analysis of geographic patterns in service requests
        
        Args:
            df: DataFrame with 311 service request data
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with insights on geographic patterns
        """
        # Find zipcode column
        zipcode_col = next((col for col in df.columns if 'zip' in col.lower()), None)
        if zipcode_col is None or zipcode_col not in df.columns:
            return "Could not find zipcode column in the data."
        
        # Service request type column
        sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
        if sr_type_col is None:
            potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
            sr_type_col = next((col for col in potential_cols if col in df.columns), None)
        
        # Collect geographic stats
        geo_stats = {}
        
        # Top zipcodes by request volume
        top_zipcodes = df[zipcode_col].value_counts().head(10).to_dict()
        geo_stats['top_zipcodes_by_volume'] = top_zipcodes
        
        # Performance by zipcode
        if 'performance_ratio' in df.columns and 'is_resolved' in df.columns:
            zipcode_performance = {}
            for zipcode in top_zipcodes.keys():
                zipcode_df = df[df[zipcode_col] == zipcode]
                if zipcode_df['is_resolved'].sum() > 0:
                    avg_perf = zipcode_df.loc[zipcode_df['is_resolved'], 'performance_ratio'].mean()
                    meeting_target = (zipcode_df.loc[zipcode_df['is_resolved'], 'performance_ratio'] <= 1.1).mean() * 100
                    zipcode_performance[zipcode] = {
                        'avg_performance_ratio': float(avg_perf),
                        'percent_meeting_target': float(meeting_target)
                    }
            
            geo_stats['zipcode_performance'] = zipcode_performance
        
        # Request types by zipcode
        if sr_type_col and sr_type_col in df.columns:
            zipcode_top_types = {}
            for zipcode in top_zipcodes.keys():
                zipcode_df = df[df[zipcode_col] == zipcode]
                top_types = zipcode_df[sr_type_col].value_counts().head(5).to_dict()
                zipcode_top_types[zipcode] = top_types
            
            geo_stats['top_request_types_by_zipcode'] = zipcode_top_types
        
        # Create the prompt - use custom encoder for NumPy types
        stats_json = json.dumps(geo_stats, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are geographic statistics for Miami-Dade County's 311 service requests.
        
        {stats_json}
        
        Analyze these statistics and provide insights on:
        1. Geographic patterns in service request volume
        2. Geographic disparities in service request resolution times
        3. How request types vary across different areas
        4. Socioeconomic or infrastructure factors that might explain these patterns
        5. Recommendations for addressing geographic disparities in service delivery
        
        Focus on actionable insights that would help service managers improve equitable service delivery across all areas.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst and public service expert specializing in geographic analysis of 311 service requests."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing geographic patterns: {str(e)}"
    
    def generate_improvement_plan(self, metrics: Dict, max_tokens: int = 1500) -> str:
        """
        Generate an improvement plan based on performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with a detailed improvement plan
        """
        # Create a prompt with the metrics - use custom encoder for NumPy types
        metrics_json = json.dumps(metrics, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are performance metrics for Miami-Dade County 311 service requests.
        
        {metrics_json}
        
        Based on these metrics, develop a comprehensive improvement plan that addresses:
        
        1. PRIORITY ISSUES: Identify the top 3-5 service request types or areas that need the most improvement
        
        2. ROOT CAUSES: For each priority issue, analyze potential root causes for performance gaps
        
        3. ACTION PLAN: For each priority issue, recommend specific actions to improve performance, including:
           - Process improvements
           - Resource allocation changes
           - Technology improvements
           - Training needs
           - Policy changes
        
        4. IMPLEMENTATION TIMELINE: Suggest a reasonable timeline for implementing these improvements
        
        5. KEY PERFORMANCE INDICATORS: Specify how to measure success for each improvement
        
        Create a practical, actionable plan that city managers could implement to improve service delivery.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a public service management consultant specializing in 311 system optimization and municipal service improvement."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating improvement plan: {str(e)}"

    def perform_seasonal_analysis(self, df: pd.DataFrame, max_tokens: int = 1000) -> str:
        """
        Analyze seasonal patterns in service requests
        
        Args:
            df: DataFrame with 311 service request data
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with insights on seasonal patterns
        """
        # Find date column
        created_date_col = next((col for col in df.columns if 'creat' in col.lower() and 'date' in col.lower()), None)
        if created_date_col is None:
            potential_cols = ['CREATED_DATE', 'CREATION_DATE', 'OPEN_DATE', 'CREATEDATE', 'SR_CREATE_DATE']
            created_date_col = next((col for col in potential_cols if col in df.columns), None)
        
        if created_date_col is None or created_date_col not in df.columns:
            return "Could not find creation date column in the data."
        
        # SR type column
        sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
        if sr_type_col is None:
            potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE', 'ISSUETYPE', 'Category']
            sr_type_col = next((col for col in potential_cols if col in df.columns), None)
        
        # Parse dates
        df[created_date_col] = pd.to_datetime(df[created_date_col], errors='coerce')
        
        # Prepare seasonal stats
        seasonal_stats = {}
        
        # Add month and season
        df['month'] = df[created_date_col].dt.month
        df['season'] = pd.cut(
            df['month'], 
            bins=[0, 3, 6, 9, 12], 
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        )
        
        # Monthly request volumes
        monthly_counts = df.groupby('month').size()
        seasonal_stats['monthly_volumes'] = {int(k): int(v) for k, v in monthly_counts.items()}
        
        # Seasonal request volumes
        seasonal_counts = df.groupby('season').size()
        seasonal_stats['seasonal_volumes'] = {str(k): int(v) for k, v in seasonal_counts.items()}
        
        # Request types by season
        if sr_type_col and sr_type_col in df.columns:
            seasonal_types = {}
            for season in df['season'].unique():
                if pd.isna(season):
                    continue
                
                season_df = df[df['season'] == season]
                top_types = season_df[sr_type_col].value_counts().head(5).to_dict()
                seasonal_types[str(season)] = {str(k): int(v) for k, v in top_types.items()}
            
            seasonal_stats['top_request_types_by_season'] = seasonal_types
        
        # Resolution performance by season
        if 'performance_ratio' in df.columns and 'is_resolved' in df.columns:
            seasonal_performance = {}
            for season in df['season'].unique():
                if pd.isna(season):
                    continue
                
                season_df = df[df['season'] == season]
                if season_df['is_resolved'].sum() > 0:
                    avg_perf = season_df.loc[season_df['is_resolved'], 'performance_ratio'].mean()
                    meeting_target = (season_df.loc[season_df['is_resolved'], 'performance_ratio'] <= 1.1).mean() * 100
                    seasonal_performance[str(season)] = {
                        'avg_performance_ratio': float(avg_perf),
                        'percent_meeting_target': float(meeting_target)
                    }
            
            seasonal_stats['seasonal_performance'] = seasonal_performance
        
        # Create the prompt - use custom encoder for NumPy types
        stats_json = json.dumps(seasonal_stats, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are seasonal statistics for Miami-Dade County's 311 service requests.
        
        {stats_json}
        
        Analyze these statistics and provide insights on:
        1. How service request volumes vary throughout the year
        2. How service request types change seasonally
        3. Whether service performance varies by season
        4. How these seasonal patterns might impact resource planning
        5. Recommendations for addressing seasonal fluctuations in service demand
        
        Focus on actionable insights that would help service managers plan and allocate resources efficiently throughout the year.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst and public service expert specializing in seasonal analysis of 311 service requests."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing seasonal patterns: {str(e)}"

    def comparative_analysis(self, df: pd.DataFrame, comparison_type: str = "priority", max_tokens: int = 1500) -> str:
        """
        Perform advanced comparative analysis between different categories of service requests
        
        Args:
            df: DataFrame with 311 service request data
            comparison_type: Type of comparison to perform ("priority", "request_type", "geographic", "method")
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with detailed comparative analysis
        """
        comparison_data = {}
        
        if comparison_type == "priority":
            # Compare performance across priority levels
            if 'PRIORITY' not in df.columns:
                return "Priority data not available in the dataset."
            
            comparison_data['type'] = "priority"
            comparison_data['categories'] = {}
            
            # Get unique priority levels
            priorities = df['PRIORITY'].unique()
            
            for priority in priorities:
                priority_df = df[df['PRIORITY'] == priority]
                metrics = {
                    'count': len(priority_df),
                    'percent_of_total': len(priority_df) / len(df) * 100
                }
                
                if 'resolution_hours' in priority_df.columns:
                    metrics['avg_resolution_hours'] = priority_df['resolution_hours'].mean()
                    metrics['median_resolution_hours'] = priority_df['resolution_hours'].median()
                
                if 'is_resolved' in priority_df.columns:
                    metrics['resolution_rate'] = priority_df['is_resolved'].mean() * 100
                
                if 'performance_ratio' in priority_df.columns:
                    metrics['avg_performance_ratio'] = priority_df['performance_ratio'].mean()
                    metrics['percent_meeting_target'] = (priority_df['performance_ratio'] <= 1.1).mean() * 100
                
                comparison_data['categories'][str(priority)] = metrics
            
        elif comparison_type == "request_type":
            # Compare performance across request types
            sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
            if sr_type_col is None:
                potential_cols = ['SR_TYPE', 'SERVICE_TYPE', 'REQUEST_TYPE', 'CASE_TYPE']
                sr_type_col = next((col for col in potential_cols if col in df.columns), None)
            
            if sr_type_col is None:
                return "Service request type data not available in the dataset."
            
            comparison_data['type'] = "request_type"
            comparison_data['categories'] = {}
            
            # Get top 10 request types by volume
            top_types = df[sr_type_col].value_counts().head(10).index
            
            for req_type in top_types:
                type_df = df[df[sr_type_col] == req_type]
                metrics = {
                    'count': len(type_df),
                    'percent_of_total': len(type_df) / len(df) * 100
                }
                
                if 'resolution_hours' in type_df.columns:
                    metrics['avg_resolution_hours'] = type_df['resolution_hours'].mean()
                    metrics['median_resolution_hours'] = type_df['resolution_hours'].median()
                
                if 'is_resolved' in type_df.columns:
                    metrics['resolution_rate'] = type_df['is_resolved'].mean() * 100
                
                if 'performance_ratio' in type_df.columns:
                    metrics['avg_performance_ratio'] = type_df['performance_ratio'].mean()
                    metrics['percent_meeting_target'] = (type_df['performance_ratio'] <= 1.1).mean() * 100
                
                comparison_data['categories'][str(req_type)] = metrics
                
        elif comparison_type == "geographic":
            # Compare performance across geographic areas
            zipcode_col = next((col for col in df.columns if 'zip' in col.lower()), None)
            if zipcode_col is None:
                return "Zipcode data not available in the dataset."
            
            comparison_data['type'] = "geographic"
            comparison_data['categories'] = {}
            
            # Get top 10 zipcodes by volume
            top_zipcodes = df[zipcode_col].value_counts().head(10).index
            
            for zipcode in top_zipcodes:
                zipcode_df = df[df[zipcode_col] == zipcode]
                metrics = {
                    'count': len(zipcode_df),
                    'percent_of_total': len(zipcode_df) / len(df) * 100
                }
                
                if 'resolution_hours' in zipcode_df.columns:
                    metrics['avg_resolution_hours'] = zipcode_df['resolution_hours'].mean()
                    metrics['median_resolution_hours'] = zipcode_df['resolution_hours'].median()
                
                if 'is_resolved' in zipcode_df.columns:
                    metrics['resolution_rate'] = zipcode_df['is_resolved'].mean() * 100
                
                if 'performance_ratio' in zipcode_df.columns:
                    metrics['avg_performance_ratio'] = zipcode_df['performance_ratio'].mean()
                    metrics['percent_meeting_target'] = (zipcode_df['performance_ratio'] <= 1.1).mean() * 100
                
                # Get top request types for this zipcode
                if sr_type_col and sr_type_col in zipcode_df.columns:
                    metrics['top_request_types'] = zipcode_df[sr_type_col].value_counts().head(3).to_dict()
                
                comparison_data['categories'][str(zipcode)] = metrics
                
        elif comparison_type == "method":
            # Compare performance across method received
            if 'METHOD_RECEIVED' not in df.columns:
                return "Method received data not available in the dataset."
            
            comparison_data['type'] = "method"
            comparison_data['categories'] = {}
            
            # Get unique methods
            methods = df['METHOD_RECEIVED'].unique()
            
            for method in methods:
                method_df = df[df['METHOD_RECEIVED'] == method]
                metrics = {
                    'count': len(method_df),
                    'percent_of_total': len(method_df) / len(df) * 100
                }
                
                if 'resolution_hours' in method_df.columns:
                    metrics['avg_resolution_hours'] = method_df['resolution_hours'].mean()
                    metrics['median_resolution_hours'] = method_df['resolution_hours'].median()
                
                if 'is_resolved' in method_df.columns:
                    metrics['resolution_rate'] = method_df['is_resolved'].mean() * 100
                
                if 'performance_ratio' in method_df.columns:
                    metrics['avg_performance_ratio'] = method_df['performance_ratio'].mean()
                    metrics['percent_meeting_target'] = (method_df['performance_ratio'] <= 1.1).mean() * 100
                
                comparison_data['categories'][str(method)] = metrics
        
        # Create the prompt
        comparison_json = json.dumps(comparison_data, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below are comparative metrics for different categories of 311 service requests in Miami-Dade County.
        
        {comparison_json}
        
        Perform a detailed comparative analysis:
        1. Compare performance metrics across the different categories
        2. Identify which categories perform significantly better or worse than others
        3. Analyze potential reasons for the performance differences
        4. Highlight the specific strengths and weaknesses of each category
        5. Recommend targeted improvements for the underperforming categories
        6. Identify best practices from the high-performing categories that could be applied elsewhere
        
        Format your response with clear section headers and bullet points for key insights.
        Focus on actionable insights that would help managers understand performance variations and make targeted improvements.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst and public service expert specializing in comparative analysis of 311 service request performance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error performing comparative analysis: {str(e)}"
    
    def analyze_custom_question(self, df: pd.DataFrame, question: str, metrics: Dict = None, max_tokens: int = 1500) -> str:
        """
        Answer a custom question about the 311 service request data
        
        Args:
            df: DataFrame with 311 service request data
            question: The specific question to analyze
            metrics: Optional performance metrics dictionary
            max_tokens: Maximum tokens for OpenAI response
            
        Returns:
            String with analysis answering the custom question
        """
        # Prepare data summary for the model
        data_summary = {
            "dataset_size": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "column_list": list(df.columns),
            "data_sample": df.head(5).to_dict(orient='records')
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            data_summary["numeric_stats"] = {}
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                try:
                    data_summary["numeric_stats"][col] = {
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
                except:
                    pass
        
        # Add categorical column distributions
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            data_summary["categorical_distributions"] = {}
            for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                try:
                    value_counts = df[col].value_counts().head(5).to_dict()
                    data_summary["categorical_distributions"][col] = value_counts
                except:
                    pass
        
        # Add metrics if provided
        if metrics:
            data_summary["performance_metrics"] = metrics
        
        # Create the prompt
        data_json = json.dumps(data_summary, indent=2, cls=NumpyEncoder)
        
        prompt = f"""
        Below is a summary of the Miami-Dade County 311 service request dataset.
        
        {data_json}
        
        Please answer the following question about this dataset:
        
        QUESTION: {question}
        
        Analyze the data provided and answer the question thoroughly. If the data summary doesn't contain enough information to fully answer the question, explain what additional data would be needed.
        
        Format your response with clear section headers and bullet points for key insights.
        Focus on actionable insights that would help managers understand and improve service delivery.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data scientist specializing in public service analytics. Answer questions about 311 service request data thoroughly and accurately based on the provided data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing custom question: {str(e)}" 