import os
import pandas as pd
import ssl
import urllib.request
import requests
from io import StringIO
import streamlit as st
import kaggle

# Create an unverified SSL context for macOS
ssl_context = ssl._create_unverified_context()

def safe_read_csv_from_url(url):
    """
    Safely read CSV from a URL using an unverified SSL context
    
    Args:
        url: URL to read CSV from
        
    Returns:
        Pandas DataFrame or None if there was an error
    """
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            csv_data = response.read().decode('utf-8')
            return pd.read_csv(StringIO(csv_data))
    except Exception as e:
        st.error(f"Error reading from URL: {str(e)}")
        # Fallback to local file if it exists
        if os.path.exists("data/auto-mpg.csv"):
            return pd.read_csv("data/auto-mpg.csv")
        return None

@st.cache_data
def get_miami_311_data(limit=1000):
    """
    Fetches Miami-Dade County 311 Service Requests data
    
    Args:
        limit: Number of records to fetch (default 1000)
        
    Returns:
        Pandas DataFrame with the 311 service request data
    """
    try:
        # Direct CSV download URL
        csv_url = "https://opendata.arcgis.com/api/v3/datasets/7cc10915ede14bb58be312413842a4ce_0/downloads/data?format=csv&spatialRefId=4326&where=1=1"
        
        # Try to download directly
        df = safe_read_csv_from_url(csv_url)
        
        # If direct download fails, try alternative API endpoint
        if df is None:
            # ArcGIS REST API endpoint with query
            api_url = f"https://services.arcgis.com/8Pc9XBTAsYuxx9Ny/arcgis/rest/services/311_Service_Requests/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json&resultRecordCount={limit}"
            
            # Use requests with SSL verification disabled
            response = requests.get(api_url, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract features and flatten attributes
                records = []
                for feature in data.get('features', []):
                    attributes = feature.get('attributes', {})
                    # Add geometry if available
                    if 'geometry' in feature:
                        attributes['longitude'] = feature['geometry'].get('x')
                        attributes['latitude'] = feature['geometry'].get('y')
                    records.append(attributes)
                
                df = pd.DataFrame(records)
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                return None
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching Miami 311 data: {str(e)}")
        return None

@st.cache_data
def download_kaggle_dataset(dataset_name):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset name in format 'username/dataset-name'
        
    Returns:
        Pandas DataFrame with the dataset
    """
    try:
        # Set Kaggle credentials - try secrets first, then environment variables, then file
        kaggle_username = st.secrets.get("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
        kaggle_key = st.secrets.get("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        
        # Check if the kaggle.json file exists
        kaggle_file_exists = os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))
        
        if not kaggle_username and not kaggle_key and not kaggle_file_exists:
            st.warning("Kaggle credentials not found. Using pre-downloaded dataset.")
            # Fallback to a specific car dataset if credentials aren't available
            if os.path.exists("data/auto-mpg.csv"):
                return pd.read_csv("data/auto-mpg.csv")
            return safe_read_csv_from_url("https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv")
        
        # Download the dataset
        kaggle.api.authenticate()
        
        # Create a data directory if it doesn't exist
        if not os.path.exists("./data"):
            os.makedirs("./data")
        
        kaggle.api.dataset_download_files(dataset_name, path="./data", unzip=True)
        
        # Find the CSV file in the downloaded data directory
        for file in os.listdir("./data"):
            if file.endswith(".csv"):
                df = pd.read_csv(f"./data/{file}")
                return df
        
        st.error("No CSV file found in the downloaded dataset.")
        return None
    except Exception as e:
        st.error(f"Error downloading Kaggle dataset: {str(e)}")
        # Fallback to a specific car dataset
        if os.path.exists("data/auto-mpg.csv"):
            return pd.read_csv("data/auto-mpg.csv")
        return safe_read_csv_from_url("https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv")

def preprocess_311_data(df):
    """
    Preprocesses the 311 data for analysis, handling key fields and common data issues
    
    Args:
        df: DataFrame with 311 data
        
    Returns:
        Preprocessed DataFrame
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle date columns - convert to datetime
    date_columns = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        except:
            pass
    
    # Standardize column names for priority if they exist
    priority_columns = [col for col in df_clean.columns if 'priority' in col.lower()]
    if priority_columns:
        # Use the first priority column found
        priority_col = priority_columns[0]
        # Create a standardized priority column
        df_clean['PRIORITY'] = df_clean[priority_col]
    else:
        # Try to infer priority from other fields if no explicit priority column
        try:
            # Some systems encode priority in the SR_TYPE or through flags like EMERGENCY
            emergency_cols = [col for col in df_clean.columns if 'emergency' in col.lower() or 'urgent' in col.lower()]
            if emergency_cols:
                # If there are emergency columns, use them to set priority
                df_clean['PRIORITY'] = 'NORMAL'
                for col in emergency_cols:
                    # Set high priority for emergency cases
                    df_clean.loc[df_clean[col].astype(str).str.upper().isin(['YES', 'Y', 'TRUE', '1']), 'PRIORITY'] = 'HIGH'
            else:
                # Default priority if we can't infer it
                df_clean['PRIORITY'] = 'NORMAL'
        except:
            df_clean['PRIORITY'] = 'NORMAL'
    
    # Handle the method received column (how the request was submitted)
    method_columns = [col for col in df_clean.columns if 'method' in col.lower() or 'source' in col.lower() or 'channel' in col.lower()]
    if method_columns:
        # Use the first method column found
        method_col = method_columns[0]
        # Create a standardized method column
        df_clean['METHOD_RECEIVED'] = df_clean[method_col]
    else:
        # Default method if we can't find it
        df_clean['METHOD_RECEIVED'] = 'UNKNOWN'
    
    # Clean up the priority values to standardized format
    if 'PRIORITY' in df_clean.columns:
        # Convert to string
        df_clean['PRIORITY'] = df_clean['PRIORITY'].astype(str).str.upper()
        
        # Map various priority values to standard ones
        priority_map = {
            '1': 'HIGH',
            '2': 'MEDIUM',
            '3': 'NORMAL',
            '4': 'LOW',
            '5': 'LOW',
            'HIGH': 'HIGH',
            'MEDIUM': 'MEDIUM',
            'NORMAL': 'NORMAL',
            'LOW': 'LOW',
            'URGENT': 'HIGH',
            'EMERGENCY': 'HIGH',
            'CRITICAL': 'HIGH'
        }
        
        # Apply mapping for known values, default to NORMAL for unknown
        df_clean['PRIORITY'] = df_clean['PRIORITY'].apply(
            lambda x: priority_map.get(x, 'NORMAL') if x in priority_map else 'NORMAL'
        )
    
    # Clean up method received values
    if 'METHOD_RECEIVED' in df_clean.columns:
        df_clean['METHOD_RECEIVED'] = df_clean['METHOD_RECEIVED'].astype(str).str.upper()
        
        # Map various method values to standard ones
        method_map = {
            'PHONE': 'PHONE',
            'WEB': 'WEB',
            'APP': 'MOBILE APP',
            'MOBILE': 'MOBILE APP',
            'MOBILE APP': 'MOBILE APP',
            'EMAIL': 'EMAIL',
            'WALK-IN': 'WALK-IN',
            'WALKIN': 'WALK-IN',
            'IN-PERSON': 'WALK-IN',
            'COUNTER': 'WALK-IN',
            'FAX': 'OTHER',
            'MAIL': 'OTHER',
            'OTHER': 'OTHER'
        }
        
        # Apply mapping, standardizing methods for analysis
        for key in method_map:
            df_clean.loc[df_clean['METHOD_RECEIVED'].str.contains(key), 'METHOD_RECEIVED'] = method_map[key]
        
        # Set unknown for anything not mapped
        df_clean.loc[~df_clean['METHOD_RECEIVED'].isin(method_map.values()), 'METHOD_RECEIVED'] = 'UNKNOWN'
    
    # Clean up latitude and longitude for mapping
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        # Convert to numeric, coercing errors to NaN
        df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
        df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
        
        # Filter out extreme values and NaN
        mask = (
            (df_clean['latitude'] > 25.0) & (df_clean['latitude'] < 26.5) &
            (df_clean['longitude'] > -81.0) & (df_clean['longitude'] < -80.0)
        )
        
        # Tag rows with valid coordinates for mapping
        df_clean['has_valid_coords'] = mask
    
    # Drop rows with all missing values
    df_clean = df_clean.dropna(how='all')
    
    # Handle potential specific issues in 311 data
    # Fill missing service request types
    sr_type_col = next((col for col in df_clean.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
    if sr_type_col and sr_type_col in df_clean.columns:
        df_clean[sr_type_col] = df_clean[sr_type_col].fillna('UNKNOWN')
    
    return df_clean 