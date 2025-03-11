#!/usr/bin/env python3
"""
Runner script for Miami 311 Service Request Analysis application.
This script sets up the Python path and launches the Streamlit application.
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit application."""
    print("Starting Miami 311 Service Request Analysis Application...")
    
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the current directory is in the Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try to find the streamlit executable
    streamlit_path = None
    
    # Common places to look for streamlit
    potential_paths = [
        os.path.expanduser("~/Library/Python/3.13/bin/streamlit"),
        os.path.expanduser("~/Library/Python/3.12/bin/streamlit"),
        os.path.expanduser("~/Library/Python/3.11/bin/streamlit"),
        os.path.expanduser("~/Library/Python/3.10/bin/streamlit"),
        os.path.expanduser("~/Library/Python/3.9/bin/streamlit"),
        "streamlit",  # If it's in the PATH
    ]
    
    # Try each potential path
    for path in potential_paths:
        try:
            # Check if the executable exists and is executable
            if os.path.isfile(path) and os.access(path, os.X_OK):
                streamlit_path = path
                break
            # Or try to run it directly if it's in the PATH
            elif path == "streamlit":
                subprocess.run(["which", "streamlit"], check=True, capture_output=True)
                streamlit_path = "streamlit"
                break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    if streamlit_path:
        print(f"Using Streamlit at: {streamlit_path}")
        # Launch the application
        # Determine which script to run based on environment
    script_to_run = "super_minimal.py" if os.environ.get("STREAMLIT_SHARING") or os.environ.get("HEROKU") else "main.py"
    print(f"Running script: {script_to_run}")
    subprocess.run([streamlit_path, "run", script_to_run])
    else:
        print("Error: Could not find Streamlit. Please install it with 'pip install streamlit'.")
        sys.exit(1)

if __name__ == "__main__":
    main() 