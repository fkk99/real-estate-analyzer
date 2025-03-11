#!/usr/bin/env python3
"""
Launcher script for Real Estate Investment Analysis System (Web Only)
"""
import os
import sys
import importlib.util

def check_dependencies():
    required_packages = ['pandas', 'numpy', 'matplotlib', 'flask']
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install required packages with: pipenv install")
        return False
    
    return True

def setup_environment():
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    return True

def main():
    print("Real Estate Investment Analysis System")
    print("--------------------------------------")
    
    if not check_dependencies():
        return 1
    
    if not setup_environment():
        return 1
    
    print("Launching web UI...")
    import web_ui
    web_ui.start_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())