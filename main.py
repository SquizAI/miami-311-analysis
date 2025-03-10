import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Import our custom modules
import data_loader
import visualizations
import analysis
import performance_analysis

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="311 Service Request Analysis",
    page_icon="📊",
    layout="wide"
)

def main():
    # Title and introduction
    st.title("📞 Miami 311 Service Request Analysis")
    st.markdown("""
    This application analyzes Miami-Dade County 311 service requests to determine whether they are meeting 
    resolution time goals. The analysis includes priority levels, method received, and detailed performance metrics.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Get API key from environment or let user input it
    openai_api_key = os.getenv("OPENAI_API_KEY")
    user_api_key = st.sidebar.text_input("OpenAI API Key (optional for AI insights)", value="", type="password")
    if user_api_key:
        openai_api_key = user_api_key
    
    # Data loading section
    st.sidebar.header("Data Options")
    record_limit = st.sidebar.slider("Number of records to fetch", 100, 5000, 1000, 100)
    
    # Load button
    if st.sidebar.button("Load Miami 311 Data"):
        with st.spinner("Loading and analyzing Miami 311 service request data..."):
            # Load the data
            df = data_loader.get_miami_311_data(limit=record_limit)
            
            if df is not None:
                # Preprocess the data to standardize columns and clean values
                df = data_loader.preprocess_311_data(df)
                
                # Perform performance analysis
                df_analyzed, metrics = analysis.analyze_311_data(df, openai_api_key)
                
                # Store in session state
                st.session_state.df = df_analyzed
                st.session_state.metrics = metrics
                
                st.success(f"✅ Successfully loaded and analyzed {len(df_analyzed)} Miami 311 Service Requests!")
            else:
                st.error("Failed to load Miami 311 data. Please try again.")
    
    # Main content
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        metrics = st.session_state.metrics
        
        # Overview section
        st.header("📋 Data Overview")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", f"{metrics['total_requests']:,}")
        with col2:
            if 'resolved_requests' in metrics:
                st.metric("Resolved Requests", f"{metrics['resolved_requests']:,}")
        with col3:
            if 'percent_meeting_target' in metrics:
                st.metric("Meeting Target", f"{metrics['percent_meeting_target']:.1f}%")
        
        # Display data sample
        with st.expander("Preview Data Sample"):
            st.dataframe(df.head())
        
        # PROMINENT QUESTION SECTION
        st.header("🔍 Ask Questions About the Data")
        st.markdown("""
        Use AI to analyze the data and get detailed insights. Ask specific questions about the 311 service 
        requests or get recommendations for improvements.
        """)
        
        # Add a quick question selection with more advanced options
        question_type = st.radio(
            "Choose analysis type:",
            ["Custom Question", "Performance Analysis", "Comparative Analysis", "Geographic Patterns", 
             "Request Type Analysis", "Seasonal Trends", "Improvement Plan"],
            horizontal=True
        )
        
        if question_type == "Custom Question":
            question = st.text_input(
                "Enter your question about the 311 data:",
                "What insights can you provide about resolution times for different priority levels?"
            )
        elif question_type == "Performance Analysis":
            question = "Analyze the overall performance in meeting resolution time targets. Which areas need improvement?"
            analysis_type = "performance"
        elif question_type == "Comparative Analysis":
            comparison_options = ["priority", "request_type", "geographic", "method"]
            comparison_names = {
                "priority": "Priority Levels", 
                "request_type": "Request Types", 
                "geographic": "Geographic Areas", 
                "method": "Request Methods"
            }
            
            selected_comparison = st.selectbox(
                "Compare performance across:", 
                options=comparison_options,
                format_func=lambda x: comparison_names[x]
            )
            
            question = f"Compare performance across different {comparison_names[selected_comparison].lower()}"
            analysis_type = f"comparative_{selected_comparison}"
        elif question_type == "Geographic Patterns":
            question = "What geographic patterns exist in service requests? Are there areas with higher concentrations of issues?"
            analysis_type = "geographic"
        elif question_type == "Request Type Analysis":
            sr_type_col = next((col for col in df.columns if 'type' in col.lower() and 'sr' in col.lower()), None)
            if sr_type_col and sr_type_col in df.columns:
                top_types = df[sr_type_col].value_counts().head(10).index.tolist()
                selected_type = st.selectbox("Select request type to analyze:", top_types)
                question = f"Analyze performance for {selected_type} requests. Are they meeting resolution targets?"
            else:
                question = "What are the most common request types and how well are they being handled?"
        elif question_type == "Seasonal Trends":
            question = "Analyze seasonal patterns in service requests. Are there specific times of year with higher volumes or different types of requests?"
            analysis_type = "seasonal"
        else:  # Improvement Plan
            question = "Create a comprehensive improvement plan with specific recommendations for enhancing service delivery."
            analysis_type = "improvement"
        
        # Show the input field for custom questions
        if question_type == "Custom Question":
            question = st.text_input("Enter your question:", question)
            analysis_type = None
            custom_question = question
        else:
            # Just display the predefined question
            st.info(f"**Question:** {question}")
            custom_question = None
        
        # Analysis button
        if st.button("Analyze Data", key="main_analyze"):
            if not openai_api_key:
                st.warning("⚠️ Please provide an OpenAI API key in the sidebar to enable AI insights.")
            else:
                with st.spinner("Analyzing data with AI... This may take a minute."):
                    # For request type analysis, get specific insights for that type
                    if question_type == "Request Type Analysis" and 'selected_type' in locals():
                        insights = analysis.generate_ai_insights(df, metrics, openai_api_key, selected_type)
                        if 'request_type_analysis' in insights:
                            st.markdown(insights['request_type_analysis'])
                        else:
                            st.error("Could not generate insights for this request type.")
                    # For custom questions
                    elif question_type == "Custom Question" and custom_question:
                        insights = analysis.generate_ai_insights(df, metrics, openai_api_key, 
                                                              custom_question=custom_question)
                        if 'custom_analysis' in insights:
                            st.markdown(insights['custom_analysis'])
                        else:
                            st.error("Could not analyze your question.")
                    # For specific analysis types
                    elif analysis_type:
                        insights = analysis.generate_ai_insights(df, metrics, openai_api_key, 
                                                             analysis_type=analysis_type)
                        # Find the first key in insights
                        if insights and len(insights) > 0:
                            key = list(insights.keys())[0]
                            st.markdown(insights[key])
                        else:
                            st.error("Could not generate the requested analysis.")
                    # Fallback
                    else:
                        st.error("Please select a valid analysis type.")
        
        # Tabs for different analysis views
        st.header("📊 Detailed Analysis")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Priority Analysis", 
            "📱 Method Analysis",
            "🗺️ Geographic Analysis", 
            "⏱️ Time Analysis",
            "🤖 AI Insights"
        ])
        
        # Tab 1: Priority Analysis
        with tab1:
            st.header("Priority Level Analysis")
            st.markdown("""
            This section analyzes whether service requests are meeting their resolution time goals 
            based on priority level. High priority tickets typically have shorter target resolution times.
            """)
            
            # Priority distribution
            visualizations.plot_priority_distribution(df)
            
            # Priority performance heatmap
            visualizations.plot_priority_performance_heatmap(df)
            
            # Priority by type
            st.subheader("Priority Distribution by Request Type")
            visualizations.plot_priority_by_type(df)
            
            # Priority performance analysis
            st.subheader("Performance Analysis by Priority and Type")
            visualizations.plot_performance_by_priority_type(df)
            
            # Priority goals analysis
            priority_goals = analysis.analyze_priority_goals(df)
            if priority_goals is not None:
                st.subheader("Goal Achievement by Priority")
                st.dataframe(priority_goals)
        
        # Tab 2: Method Analysis
        with tab2:
            st.header("Method Received Analysis")
            st.markdown("""
            This section analyzes how service requests are received (phone, web, app, etc.) and
            whether the method impacts resolution times and goal achievement.
            """)
            
            # Method distribution
            visualizations.plot_method_received_distribution(df)
            
            # Method performance analysis
            visualizations.plot_performance_by_method(df)
            
            # Method & priority analysis
            st.subheader("Priority Distribution by Method")
            visualizations.plot_method_priority_analysis(df)
            
            # Method goals analysis
            method_goals = analysis.analyze_method_received_goals(df)
            if method_goals is not None:
                st.subheader("Goal Achievement by Method")
                st.dataframe(method_goals)
        
        # Tab 3: Geographic Analysis
        with tab3:
            st.header("Geographic Analysis")
            st.markdown("""
            This map shows the geographic distribution of service requests, with colors
            indicating priority levels. This helps identify areas with high concentrations
            of specific service needs or priority levels.
            """)
            
            # Map by priority
            visualizations.plot_map_by_priority(df)
            
            # Additional geographic analysis
            if 'zipcode_metrics' in metrics:
                st.subheader("Performance by Zipcode")
                
                # Convert to DataFrame for display
                zipcode_df = pd.DataFrame.from_dict(
                    metrics['zipcode_metrics'], 
                    orient='index'
                ).reset_index()
                zipcode_df.columns = ['Zipcode'] + list(zipcode_df.columns[1:])
                
                # Display
                st.dataframe(zipcode_df)
        
        # Tab 4: Time Analysis
        with tab4:
            st.header("Time-based Analysis")
            st.markdown("""
            This section analyzes performance trends over time, including seasonal patterns
            and changes in resolution times and goal achievement.
            """)
            
            # Performance over time
            monthly_metrics = analysis.analyze_performance_over_time(df)
            if monthly_metrics is not None:
                # Add time trend visualizations directly in the visualizations module
                
                # Plot request volume over time
                st.subheader("Service Request Volume Over Time")
                fig = px.line(
                    monthly_metrics, 
                    x='month_date', 
                    y='request_count',
                    title='Monthly Service Request Volume'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot performance over time
                st.subheader("Performance Metrics Over Time")
                fig = px.line(
                    monthly_metrics, 
                    x='month_date', 
                    y=['target_met_pct', 'high_priority_pct'],
                    title='Monthly Performance Metrics',
                    labels={
                        'target_met_pct': 'Target Met %',
                        'high_priority_pct': 'High Priority %',
                        'month_date': 'Month'
                    }
                )
                # Add reference line at 80%
                fig.add_shape(
                    type="line",
                    x0=monthly_metrics['month_date'].min(),
                    x1=monthly_metrics['month_date'].max(),
                    y0=80,
                    y1=80,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed metrics as a table
                st.subheader("Monthly Performance Metrics Table")
                display_metrics = monthly_metrics.copy()
                # Format the month for readability
                display_metrics['month'] = display_metrics['month_date'].dt.strftime('%b %Y')
                # Select and order columns
                cols_to_display = ['month', 'request_count', 'target_met_pct', 'avg_resolution_hours', 'high_priority_pct']
                display_cols = [col for col in cols_to_display if col in display_metrics.columns]
                st.dataframe(display_metrics[display_cols])
        
        # Tab 5: AI Insights
        with tab5:
            st.header("Advanced AI-Powered Insights")
            st.markdown("""
            This section allows you to use AI to generate deeper insights from the 311 service request data.
            You can perform comparative analysis, ask custom questions, or generate specialized reports.
            """)
            
            if not openai_api_key:
                st.warning("Please provide an OpenAI API key in the sidebar to enable AI insights.")
            else:
                # Advanced AI analysis options
                ai_analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Comparative Analysis", "Custom Question", "Performance Deep Dive", 
                     "Improvement Recommendations", "Geographical Analysis", "Seasonal Patterns"]
                )
                
                if ai_analysis_type == "Comparative Analysis":
                    st.subheader("Compare Performance Across Categories")
                    st.markdown("""
                    This analysis compares performance metrics across different categories to identify
                    patterns, outliers, and opportunities for improvement.
                    """)
                    
                    compare_category = st.radio(
                        "Select category to compare:",
                        ["Priority Levels", "Request Types", "Geographic Areas", "Request Methods"],
                        horizontal=True
                    )
                    
                    compare_map = {
                        "Priority Levels": "priority",
                        "Request Types": "request_type",
                        "Geographic Areas": "geographic",
                        "Request Methods": "method"
                    }
                    
                    if st.button("Generate Comparative Analysis", key="compare_button"):
                        with st.spinner("Performing comparative analysis..."):
                            insights = analysis.generate_ai_insights(
                                df, metrics, openai_api_key, 
                                analysis_type=f"comparative_{compare_map[compare_category]}"
                            )
                            
                            if 'comparative_analysis' in insights:
                                st.markdown(insights['comparative_analysis'])
                            else:
                                st.error("Could not generate comparative analysis.")
                
                elif ai_analysis_type == "Custom Question":
                    st.subheader("Ask Custom Questions")
                    st.markdown("""
                    Ask specific questions about the data and get AI-powered answers drawing from the full dataset.
                    """)
                    
                    custom_q = st.text_area(
                        "Enter your question about the 311 data:",
                        "Which service requests have the highest variation in resolution times, and what factors might explain this?"
                    )
                    
                    if st.button("Analyze Question", key="custom_q_button"):
                        if custom_q:
                            with st.spinner("Analyzing your question..."):
                                insights = analysis.generate_ai_insights(
                                    df, metrics, openai_api_key, custom_question=custom_q
                                )
                                
                                if 'custom_analysis' in insights:
                                    st.markdown(insights['custom_analysis'])
                                else:
                                    st.error("Could not analyze your question.")
                        else:
                            st.warning("Please enter a question to analyze.")
                
                elif ai_analysis_type == "Performance Deep Dive":
                    st.subheader("Performance Deep Dive")
                    st.markdown("""
                    This analysis provides a detailed examination of performance metrics, identifying
                    strengths, weaknesses, and recommendations for improvement.
                    """)
                    
                    if st.button("Generate Performance Analysis", key="performance_button"):
                        with st.spinner("Generating performance analysis..."):
                            insights = analysis.generate_ai_insights(
                                df, metrics, openai_api_key, analysis_type="performance"
                            )
                            
                            if 'performance_overview' in insights:
                                st.markdown(insights['performance_overview'])
                            else:
                                st.error("Could not generate performance analysis.")
                
                elif ai_analysis_type == "Improvement Recommendations":
                    st.subheader("Improvement Recommendations")
                    st.markdown("""
                    Get specific, actionable recommendations for improving 311 service delivery
                    based on patterns and issues identified in the data.
                    """)
                    
                    if st.button("Generate Improvement Plan", key="improvement_button"):
                        with st.spinner("Generating improvement recommendations..."):
                            insights = analysis.generate_ai_insights(
                                df, metrics, openai_api_key, analysis_type="improvement"
                            )
                            
                            if 'improvement_plan' in insights:
                                st.markdown(insights['improvement_plan'])
                            else:
                                st.error("Could not generate improvement recommendations.")
                
                elif ai_analysis_type == "Geographical Analysis":
                    st.subheader("Geographical Analysis")
                    st.markdown("""
                    This analysis examines geographic patterns in service requests, identifying
                    areas with high volumes, unique needs, or performance issues.
                    """)
                    
                    if st.button("Generate Geographic Analysis", key="geo_button"):
                        with st.spinner("Analyzing geographic patterns..."):
                            insights = analysis.generate_ai_insights(
                                df, metrics, openai_api_key, analysis_type="geographic"
                            )
                            
                            if 'geographic_analysis' in insights:
                                st.markdown(insights['geographic_analysis'])
                            else:
                                st.error("Could not generate geographic analysis.")
                
                elif ai_analysis_type == "Seasonal Patterns":
                    st.subheader("Seasonal Patterns Analysis")
                    st.markdown("""
                    This analysis identifies seasonal trends in service requests, including
                    changes in volume, type, and performance throughout the year.
                    """)
                    
                    if st.button("Generate Seasonal Analysis", key="seasonal_button"):
                        with st.spinner("Analyzing seasonal patterns..."):
                            insights = analysis.generate_ai_insights(
                                df, metrics, openai_api_key, analysis_type="seasonal"
                            )
                            
                            if 'seasonal_analysis' in insights:
                                st.markdown(insights['seasonal_analysis'])
                            else:
                                st.error("Could not generate seasonal analysis.")

if __name__ == "__main__":
    main() 