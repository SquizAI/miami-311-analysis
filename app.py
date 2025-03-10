import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from openai import OpenAI
import kaggle
import tempfile
import json
import ssl
import urllib.request
from io import StringIO
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide"
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Create an unverified SSL context for macOS
ssl_context = ssl._create_unverified_context()

# Function to safely read CSV from URL
def safe_read_csv_from_url(url):
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

# Function to download Miami 311 data
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

# Function to clean and prepare Miami 311 data
def prepare_miami_311_data(df):
    """
    Cleans and prepares the Miami 311 data for analysis
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
    
    # Drop rows with all missing values
    df_clean = df_clean.dropna(how='all')
    
    # Handle potential specific issues in 311 data
    # Fill missing service request types
    if 'SR_TYPE' in df_clean.columns:
        df_clean['SR_TYPE'] = df_clean['SR_TYPE'].fillna('Unknown')
    
    return df_clean

# Function to download Kaggle dataset
@st.cache_data
def download_kaggle_dataset(dataset_name):
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

# Function to analyze data with OpenAI
def analyze_with_openai(data_description, question):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Provide insights on the dataset."},
                {"role": "user", "content": f"Dataset info: {data_description}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing with OpenAI: {str(e)}")
        return "Unable to perform analysis. Please check your API key."

# Function to generate visualization recommendation with OpenAI
def recommend_visualization(data_description, columns):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Recommend the best visualization for this dataset."},
                {"role": "user", "content": f"Dataset info: {data_description}\n\nColumns: {columns}\n\nPlease provide a JSON response with these fields: 'chart_type' (one of: 'scatter', 'bar', 'histogram', 'box', 'line', 'heatmap'), 'x_axis', 'y_axis', 'title', 'explanation'"}
            ]
        )
        
        recommendation = response.choices[0].message.content
        
        # Extract the JSON part from the response
        try:
            import re
            json_match = re.search(r'{.*}', recommendation, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return json.loads(recommendation)
        except:
            st.warning("Could not parse OpenAI visualization recommendation as JSON. Using default visualization.")
            return {
                "chart_type": "scatter",
                "x_axis": columns[0],
                "y_axis": columns[1] if len(columns) > 1 else columns[0],
                "title": "Data Visualization",
                "explanation": "Default visualization of the dataset."
            }
    except Exception as e:
        st.error(f"Error getting visualization recommendation: {str(e)}")
        return {
            "chart_type": "scatter",
            "x_axis": columns[0],
            "y_axis": columns[1] if len(columns) > 1 else columns[0],
            "title": "Data Visualization",
            "explanation": "Default visualization of the dataset."
        }

# Main app
def main():
    st.title("📊 Data Analysis with OpenAI")
    
    # Data source selection sidebar
    st.sidebar.header("Dataset Options")
    
    # Display Kaggle credentials status
    kaggle_username = os.getenv("KAGGLE_USERNAME", "")
    kaggle_file_exists = os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))
    
    if kaggle_username or kaggle_file_exists:
        st.sidebar.success("✅ Kaggle credentials detected!")
    else:
        st.sidebar.warning("⚠️ No Kaggle credentials found. Some datasets may be unavailable.")
    
    # Option to select data source type
    data_source_type = st.sidebar.radio(
        "Choose Data Source Type",
        ["Miami 311 Data", "Car Data", "Custom Kaggle Dataset"]
    )
    
    # Load data based on selection
    if data_source_type == "Miami 311 Data":
        st.sidebar.subheader("Miami 311 Service Request Data")
        record_limit = st.sidebar.slider("Number of records to fetch", 100, 5000, 1000, 100)
        
        # Data loading
        with st.spinner("Loading Miami 311 Service Request data..."):
            df = get_miami_311_data(limit=record_limit)
            if df is not None:
                df = prepare_miami_311_data(df)
                
                # Show basic info about the dataset
                if df is not None:
                    st.success(f"✅ Successfully loaded {df.shape[0]} Miami 311 Service Requests!")
                else:
                    st.error("Failed to prepare Miami 311 data.")
    
    elif data_source_type == "Car Data":
        dataset_options = {
            "Auto MPG Dataset": "data/auto-mpg.csv" if os.path.exists("data/auto-mpg.csv") else "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv",
            "Vehicle Dataset": "eeevans/vehicle-dataset",
            "Used Cars Dataset": "austinreese/craigslist-carstrucks-data"
        }
        
        selected_dataset = st.sidebar.selectbox("Select Sample Dataset", list(dataset_options.keys()))
        dataset_value = dataset_options[selected_dataset]
        
        # Check if it's a URL, local file, or a Kaggle dataset name
        if dataset_value.startswith("http"):
            df = safe_read_csv_from_url(dataset_value)
        elif os.path.exists(dataset_value):
            df = pd.read_csv(dataset_value)
        else:
            df = download_kaggle_dataset(dataset_value)
            
    else:  # Custom Kaggle Dataset
        dataset_name = st.sidebar.text_input("Enter Kaggle Dataset Name (e.g., 'username/dataset-name')", "eeevans/vehicle-dataset")
        if st.sidebar.button("Load Dataset"):
            df = download_kaggle_dataset(dataset_name)
        else:
            df = download_kaggle_dataset("eeevans/vehicle-dataset")  # Default
    
    if df is not None:
        st.write(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Display data overview
        with st.expander("Data Preview"):
            st.dataframe(df.head())
            
            # Generate data description
            data_description = f"This is a dataset with {df.shape[0]} rows and {df.shape[1]} columns. "
            data_description += f"The columns are: {', '.join(df.columns.tolist())}. "
            
            try:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    data_description += f"Numeric columns summary: {df[numeric_cols].describe().to_string()}"
            except:
                pass
        
        # Data analysis with OpenAI
        st.header("Data Analysis with OpenAI")
        
        # Customize default question based on data source
        default_question = "What insights can you provide about this dataset?"
        if data_source_type == "Miami 311 Data":
            default_question = "What patterns do you see in the Miami 311 service requests?"
        elif data_source_type == "Car Data":
            default_question = "What insights can you provide about this car dataset?"
        
        analysis_question = st.text_input("Ask a question about the data", default_question)
        
        if st.button("Analyze"):
            with st.spinner("Analyzing data with OpenAI..."):
                analysis_result = analyze_with_openai(data_description, analysis_question)
                st.markdown(analysis_result)
        
        # Data visualization
        st.header("Data Visualization")
        
        viz_option = st.radio(
            "Choose Visualization Method",
            ["AI Recommended Visualization", "Custom Visualization"]
        )
        
        if viz_option == "AI Recommended Visualization":
            with st.spinner("Getting visualization recommendation..."):
                viz_recommendation = recommend_visualization(data_description, df.columns.tolist())
                
                st.subheader(viz_recommendation.get("title", "Data Visualization"))
                st.markdown(viz_recommendation.get("explanation", ""))
                
                chart_type = viz_recommendation.get("chart_type", "scatter")
                x_axis = viz_recommendation.get("x_axis", df.columns[0])
                y_axis = viz_recommendation.get("y_axis", df.columns[1] if len(df.columns) > 1 else df.columns[0])
                
                # Handle if recommended columns are not in the dataframe
                if x_axis not in df.columns:
                    x_axis = df.columns[0]
                if y_axis not in df.columns:
                    y_axis = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                # Create the recommended visualization
                if chart_type == "scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=viz_recommendation.get("title", "Data Visualization"))
                elif chart_type == "bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=viz_recommendation.get("title", "Data Visualization"))
                elif chart_type == "histogram":
                    fig = px.histogram(df, x=x_axis, title=viz_recommendation.get("title", "Data Visualization"))
                elif chart_type == "box":
                    fig = px.box(df, x=x_axis, y=y_axis, title=viz_recommendation.get("title", "Data Visualization"))
                elif chart_type == "line":
                    fig = px.line(df, x=x_axis, y=y_axis, title=viz_recommendation.get("title", "Data Visualization"))
                elif chart_type == "heatmap":
                    # For heatmap, we need to prepare a correlation matrix
                    numeric_df = df.select_dtypes(include=['number'])
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, title=viz_recommendation.get("title", "Correlation Heatmap"))
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=viz_recommendation.get("title", "Data Visualization"))
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Custom visualization options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Line Chart", "Heatmap"]
                )
            
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                
                if chart_type in ["Scatter Plot", "Bar Chart", "Line Chart"]:
                    x_axis = st.selectbox("X-axis", df.columns.tolist())
                    y_axis = st.selectbox("Y-axis", numeric_cols if numeric_cols else df.columns.tolist())
                elif chart_type == "Histogram":
                    x_axis = st.selectbox("Column", numeric_cols if numeric_cols else df.columns.tolist())
                    y_axis = None
                elif chart_type == "Box Plot":
                    x_axis = st.selectbox("Category", categorical_cols if categorical_cols else df.columns.tolist())
                    y_axis = st.selectbox("Value", numeric_cols if numeric_cols else df.columns.tolist())
                elif chart_type == "Heatmap":
                    x_axis = y_axis = None
            
            with col3:
                if chart_type != "Heatmap":
                    color_by = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    color_col = None if color_by == "None" else color_by
                else:
                    color_col = None
            
            # Create visualization based on user selection
            if st.button("Generate Visualization"):
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_col, title=f"Distribution of {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Heatmap":
                    # For heatmap, we use correlation of numeric columns
                    numeric_df = df.select_dtypes(include=['number'])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Need at least 2 numeric columns for a correlation heatmap.")

        # Add a map visualization for Miami 311 data if coordinates are available
        if data_source_type == "Miami 311 Data":
            if "latitude" in df.columns and "longitude" in df.columns:
                st.header("Service Request Location Map")
                # Clean coordinates by removing missing or invalid values
                map_data = df.dropna(subset=["latitude", "longitude"]).copy()
                # Filter out extreme values 
                map_data = map_data[
                    (map_data["latitude"] > 25.0) & (map_data["latitude"] < 26.5) &
                    (map_data["longitude"] > -81.0) & (map_data["longitude"] < -80.0)
                ]
                
                if not map_data.empty:
                    # Get a color column if possible
                    if "SR_TYPE" in map_data.columns:
                        color_col = "SR_TYPE"
                    else:
                        color_col = None
                    
                    # Create map
                    st.subheader("311 Service Request Locations")
                    fig = px.scatter_mapbox(
                        map_data, 
                        lat="latitude", 
                        lon="longitude", 
                        color=color_col,
                        hover_name="SR_TYPE" if "SR_TYPE" in map_data.columns else None,
                        hover_data=["SR_STATUS"] if "SR_STATUS" in map_data.columns else None,
                        zoom=10,
                        mapbox_style="open-street-map"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid coordinates found for mapping service requests.")

if __name__ == "__main__":
    main() 