#!/usr/bin/env python3
"""
Launcher script for Real Estate Investment Analysis System
This script provides a single command to start the web UI
"""
import os
import sys
import argparse
import subprocess
import webbrowser
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Launcher")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'flask']
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install required packages with: pipenv install")
        return False
    
    return True

def setup_environment():
    """Set up the environment by creating necessary directories"""
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Check if database exists, create it if not
    if not os.path.exists('real_estate_data.db'):
        logger.info("Initializing new database...")
        try:
            import sqlite3
            conn = sqlite3.connect('real_estate_data.db')
            conn.execute('''
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                address TEXT,
                purchase_date TEXT,
                purchase_price REAL,
                current_valuation REAL
            )
            ''')
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    return True

def main():
    """Main function to parse arguments and launch the web UI"""
    parser = argparse.ArgumentParser(description='Real Estate Investment Analysis System')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to run web UI on (default: 5000)')
    parser.add_argument('--debug', action='store_true', 
                     help='Enable debug mode')
    args = parser.parse_args()
    
    logger.info("Starting Real Estate Investment Analysis System")
    
    # Set debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        os.environ['DEBUG'] = 'True'
        logger.debug("Debug mode enabled")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set up environment
    if not setup_environment():
        return 1
    
    # Launch the web UI
    logger.info(f"Launching web UI on port {args.port}...")
    try:
        import web_ui
        web_ui.start_server(args.port)
    except ImportError:
        logger.error("Failed to import web_ui module")
        return 1
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Error running web UI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())