# Miami 311 Service Request Analysis

An advanced data analysis application that examines Miami-Dade County 311 service requests to determine whether they are meeting resolution time goals, with detailed analysis by priority level, method received, geographic distribution, and time trends.

## Features

- **Priority Level Analysis**: Analyze how different priority levels perform against resolution time targets
- **Method Analysis**: Examine how different methods of receiving requests (phone, web, app) impact resolution
- **Geographic Analysis**: Map service requests by location and analyze performance by area
- **Time Trends**: Analyze seasonal patterns and performance changes over time
- **AI-Powered Insights**: Leverage OpenAI to generate deeper analysis and improvement recommendations

## Components

The application has been designed with a modular architecture:

- **main.py**: Main application entry point that orchestrates all components
- **data_loader.py**: Functions for loading and preprocessing data
- **performance_analysis.py**: Core analysis of resolution times and goal achievement
- **ai_insights.py**: AI-powered advanced insights using OpenAI
- **analysis.py**: Additional analysis functions that combine the above modules
- **visualizations.py**: Visualization components
- **run.py**: Helper script to run the application

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create a `.env` file with your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

3. Run the application:
   ```
   python run.py
   ```
   or
   ```
   ./run.py
   ```

   If you prefer to run Streamlit directly:
   ```
   streamlit run main.py
   ```

## Usage

1. Use the sidebar to set configuration options:
   - Enter your OpenAI API key (optional, for AI insights)
   - Select the number of records to fetch

2. Click "Load Miami 311 Data" to fetch and analyze the data

3. Explore different analysis tabs:
   - **Priority Analysis**: See how different priority levels perform
   - **Method Analysis**: Analyze impact of different request methods
   - **Geographic Analysis**: Explore spatial patterns
   - **Time Analysis**: View trends over time
   - **AI Insights**: Get AI-powered recommendations and analysis

## Data Source

The application fetches data from the Miami-Dade County Open Data Portal:
- Dataset: 311 Service Requests
- Source URL: [Miami-Dade 311 Service Requests](https://datahub-miamigis.opendata.arcgis.com/datasets/7cc10915ede14bb58be312413842a4ce_0/explore)

## Notes

- This application includes flexible parsing to accommodate various 311 data formats
- It automatically handles SSL certificate issues on macOS
- Priority levels and method received information are standardized for consistent analysis 