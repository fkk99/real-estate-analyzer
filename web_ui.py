"""
Enhanced web-based UI for Real Estate Investment Analysis
Uses Flask framework with improved error handling and debugging
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import sys
import json
import sqlite3
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import webbrowser
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebUI")

# Try to import the analyzer
try:
    from real_estate_analyzer import RealEstateAnalyzer
    logger.info("Successfully imported RealEstateAnalyzer")
except ImportError as e:
    logger.error(f"Could not import RealEstateAnalyzer: {e}")
    logger.error("Will use subprocess instead")
    import subprocess

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'csv'}
DB_PATH = 'real_estate_data.db'
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# Create app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = 'real-estate-analyzer-secret-key'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Global variables for tracking analysis status
analysis_status = {
    'in_progress': False,
    'message': '',
    'progress': 0,
    'error': None,
    'start_time': None,
    'end_time': None,
    'debug_info': [],
    'file_name': None
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_debug_info(message, level="INFO"):
    """Add a debug message to the status info."""
    if DEBUG_MODE or level in ("ERROR", "WARNING"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        analysis_status['debug_info'].append({
            'time': timestamp,
            'message': message,
            'level': level
        })
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

@app.route('/')
def index():
    """Main page route."""
    logger.debug("Loading index page")
    
    # Get list of reports
    reports = []
    try:
        for filename in os.listdir(app.config['REPORTS_FOLDER']):
            if filename.endswith('.html'):
                file_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
                mtime = os.path.getmtime(file_path)
                date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                reports.append({
                    'filename': filename,
                    'date': date_str,
                    'path': file_path
                })
        
        # Sort by date, newest first
        reports = sorted(reports, key=lambda x: x['date'], reverse=True)
        logger.debug(f"Found {len(reports)} report files")
    except Exception as e:
        logger.error(f"Error loading reports: {e}")
        reports = []
    
    # Get properties from database if available
    properties = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, address, purchase_date, purchase_price FROM properties")
        properties = [
            {
                'name': row[0],
                'address': row[1],
                'purchase_date': row[2],
                'purchase_price': row[3]
            } for row in cursor.fetchall()
        ]
        conn.close()
        logger.debug(f"Found {len(properties)} properties in database")
    except Exception as e:
        logger.error(f"Error loading properties: {e}")
    
    # Check if categories are defined
    categories_defined = False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='transactions'")
        if cursor.fetchone()[0] > 0:
            categories_defined = True
        conn.close()
    except Exception as e:
        logger.error(f"Error checking for transaction table: {e}")
    
    return render_template('index.html', 
                          reports=reports, 
                          properties=properties,
                          categories_defined=categories_defined,
                          analysis_status=analysis_status,
                          debug_mode=DEBUG_MODE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis."""
    logger.info("File upload request received")
    
    # Check if analysis is already in progress
    if analysis_status['in_progress']:
        message = 'Analysis already in progress. Please wait.'
        logger.warning(message)
        flash(message, 'warning')
        return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        message = 'No file part in the request'
        logger.error(message)
        flash(message, 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        message = 'No selected file'
        logger.error(message)
        flash(message, 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Get form options
            save_to_db = 'save_to_db' in request.form
            open_report = 'open_report' in request.form
            
            # Start analysis in background thread
            threading.Thread(target=run_analysis, args=(file_path, save_to_db, open_report)).start()
            
            message = f'File {filename} uploaded and analysis started'
            logger.info(message)
            flash(message, 'success')
            
            # Reset debug info
            analysis_status['debug_info'] = []
            analysis_status['file_name'] = filename
            
            return redirect(url_for('index'))
        except Exception as e:
            error_msg = f"Error processing upload: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            flash(error_msg, 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.', 'error')
    return redirect(url_for('index'))

@app.route('/reports/<filename>')
def report(filename):
    """Serve a specific report file."""
    return send_from_directory(app.config['REPORTS_FOLDER'], filename)

@app.route('/status')
def status():
    """Return the current analysis status as JSON."""
    return jsonify(analysis_status)

@app.route('/clear-status', methods=['POST'])
def clear_status():
    """Clear the error status."""
    analysis_status['error'] = None
    analysis_status['debug_info'] = []
    return redirect(url_for('index'))

@app.route('/debug-mode/<setting>', methods=['POST'])
def set_debug_mode(setting):
    """Toggle debug mode setting."""
    global DEBUG_MODE
    DEBUG_MODE = (setting.lower() == 'on')
    
    # Set environment variable for child processes
    os.environ['DEBUG'] = 'True' if DEBUG_MODE else 'False'
    
    logger.info(f"Debug mode set to: {DEBUG_MODE}")
    return redirect(url_for('index'))

def run_analysis(file_path, save_to_db, open_report):
    """Run the analysis process with comprehensive logging and error handling."""
    global analysis_status
    
    # Initialize status
    analysis_status['in_progress'] = True
    analysis_status['message'] = 'Starting analysis...'
    analysis_status['progress'] = 5
    analysis_status['error'] = None
    analysis_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    analysis_status['end_time'] = None
    
    add_debug_info(f"Starting analysis of {file_path}")
    add_debug_info(f"Save to database: {save_to_db}")
    
    try:
        # Check if we can import the analyzer
        if 'RealEstateAnalyzer' in globals():
            add_debug_info("Using RealEstateAnalyzer module")
            
            # Create analyzer with debug mode if set
            analyzer = RealEstateAnalyzer(DB_PATH)
            if DEBUG_MODE:
                analyzer.set_debug_mode(True)
                add_debug_info("Debug mode enabled for analyzer")
            
            # Process CSV file
            analysis_status['message'] = 'Parsing CSV file...'
            analysis_status['progress'] = 20
            add_debug_info("Starting CSV parsing")
            df = analyzer.parse_csv(file_path)
            
            if df is not None:
                add_debug_info(f"CSV parsed successfully: {len(df)} rows")
                
                analysis_status['message'] = 'Categorizing transactions...'
                analysis_status['progress'] = 40
                add_debug_info("Starting transaction categorization")
                categorized = analyzer.categorize_transactions()
                
                if categorized is not None:
                    add_debug_info(f"Transactions categorized: {len(categorized)} rows")
                    
                    analysis_status['message'] = 'Analyzing data...'
                    analysis_status['progress'] = 60
                    add_debug_info("Starting data analysis")
                    analysis = analyzer.analyze_data()
                    
                    if analysis:
                        add_debug_info("Data analysis complete")
                        
                        # Save to database if option is selected
                        if save_to_db:
                            analysis_status['message'] = 'Saving to database...'
                            analysis_status['progress'] = 80
                            add_debug_info("Saving to database")
                            save_result = analyzer.save_to_database()
                            if save_result:
                                add_debug_info("Database save successful")
                            else:
                                add_debug_info("Database save failed", "WARNING")
                        
                        # Generate reports
                        analysis_status['message'] = 'Generating reports...'
                        analysis_status['progress'] = 90
                        add_debug_info("Generating reports")
                        report_result = analyzer.generate_reports(app.config['REPORTS_FOLDER'])
                        
                        if report_result:
                            add_debug_info("Report generation successful")
                            report_path = os.path.join(app.config['REPORTS_FOLDER'], 'real_estate_analysis_report.html')
                            
                            # Open report if option is selected
                            if open_report and os.path.exists(report_path):
                                add_debug_info("Opening report in browser")
                                webbrowser.open(f"file://{os.path.abspath(report_path)}")
                        else:
                            add_debug_info("Report generation failed", "WARNING")
                    else:
                        add_debug_info("Data analysis failed", "ERROR")
                        analysis_status['error'] = "Data analysis failed. Check logs for details."
                else:
                    add_debug_info("Transaction categorization failed", "ERROR")
                    analysis_status['error'] = "Transaction categorization failed. Check logs for details."
            else:
                add_debug_info("CSV parsing failed", "ERROR")
                analysis_status['error'] = "Failed to parse CSV file. Check format and encoding."
            
            # Close analyzer connection
            analyzer.close()
        else:
            # Use subprocess to call the script
            add_debug_info("Using subprocess to call analyzer script")
            analysis_status['message'] = 'Executing analysis script...'
            analysis_status['progress'] = 30
            
            cmd = [
                sys.executable, 
                "real_estate_analyzer.py", 
                "--input", file_path, 
                "--output", app.config['REPORTS_FOLDER']
            ]
            
            if DEBUG_MODE:
                cmd.append("--debug")
            
            if not save_to_db:
                cmd.append("--no-save")
            
            add_debug_info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            # Log output
            for line in stdout.splitlines():
                if line.strip():
                    add_debug_info(f"STDOUT: {line}")
            
            if stderr:
                for line in stderr.splitlines():
                    if line.strip():
                        add_debug_info(f"STDERR: {line}", "WARNING")
            
            if process.returncode != 0:
                error_msg = f"Analysis script failed with return code {process.returncode}"
                add_debug_info(error_msg, "ERROR")
                raise Exception(error_msg)
            
            analysis_status['progress'] = 90
            add_debug_info("Analysis script completed successfully")
            
            report_path = os.path.join(app.config['REPORTS_FOLDER'], 'real_estate_analysis_report.html')
            if open_report and os.path.exists(report_path):
                add_debug_info("Opening report in browser")
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
        
        analysis_status['message'] = 'Analysis complete!'
        analysis_status['progress'] = 100
        add_debug_info("Analysis process completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Error in analysis: {error_msg}")
        logger.error(tb)
        
        analysis_status['error'] = error_msg
        analysis_status['message'] = f"Error: {error_msg}"
        add_debug_info(f"Error: {error_msg}", "ERROR")
        add_debug_info(tb, "ERROR")
    finally:
        analysis_status['in_progress'] = False
        analysis_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Templates folder
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create index.html template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Real Estate Investment Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-top: 0;
        }
        .section {
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .progress {
            height: 20px;
            background-color: #f5f5f5;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background-color: #4CAF50;
            text-align: center;
            line-height: 20px;
            color: white;
            transition: width 0.3s;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .flash {
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .flash.success {
            background-color: #dff0d8;
            color: #3c763d;
        }
        .flash.error {
            background-color: #f2dede;
            color: #a94442;
        }
        .flash.warning {
            background-color: #fcf8e3;
            color: #8a6d3b;
        }
        button, .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        button:hover, .button:hover {
            background-color: #45a049;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        .hidden {
            display: none;
        }
        .debug-log {
            background-color: #f8f9fa;
            padding: 10px;
            font-family: monospace;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        .debug-entry {
            margin-bottom: 5px;
            padding: 3px;
        }
        .debug-entry.ERROR {
            color: #dc3545;
            font-weight: bold;
        }
        .debug-entry.WARNING {
            color: #ffc107;
        }
        .debug-controls {
            margin-top: 10px;
            text-align: right;
        }
        .debug-toggle {
            background-color: #6c757d;
        }
        .clear-button {
            background-color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real Estate Investment Analyzer</h1>
        
        {% for category, message in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ category }}">{{ message }}</div>
        {% endfor %}
        
        {% if analysis_status.file_name %}
        <div class="flash success">
            Working with file: {{ analysis_status.file_name }}
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Upload Bank Statement</h2>
            <form method="post" action="{{ url_for('upload_file') }}" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv">
                <div>
                    <label>
                        <input type="checkbox" name="save_to_db" checked> Save to database
                    </label>
                </div>
                <div>
                    <label>
                        <input type="checkbox" name="open_report" checked> Open report after analysis
                    </label>
                </div>
                <button type="submit">Upload and Analyze</button>
            </form>
        </div>
        
        <div id="status-section" class="section {% if not analysis_status.in_progress and not analysis_status.error %}hidden{% endif %}">
            <h2>Analysis Status</h2>
            <div id="status-message">{{ analysis_status.message }}</div>
            
            <div class="progress">
                <div id="progress-bar" class="progress-bar" style="width: {{ analysis_status.progress }}%">
                    {{ analysis_status.progress }}%
                </div>
            </div>
            
            {% if analysis_status.start_time %}
            <div>
                <strong>Started:</strong> {{ analysis_status.start_time }}
                {% if analysis_status.end_time %}
                <br><strong>Completed:</strong> {{ analysis_status.end_time }}
                {% endif %}
            </div>
            {% endif %}
            
            <div id="error-message" class="flash error {% if not analysis_status.error %}hidden{% endif %}">
                {{ analysis_status.error }}
                
                {% if analysis_status.error %}
                <form method="post" action="{{ url_for('clear_status') }}" style="margin-top: 10px;">
                    <button type="submit" class="clear-button">Clear Error</button>
                </form>
                {% endif %}
            </div>
            
            {% if debug_mode and analysis_status.debug_info %}
            <div>
                <h3>Debug Information</h3>
                <div class="debug-log">
                    {% for entry in analysis_status.debug_info %}
                    <div class="debug-entry {{ entry.level }}">
                        [{{ entry.time }}] {{ entry.level }}: {{ entry.message }}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            <div class="debug-controls">
                <form method="post" action="{{ url_for('set_debug_mode', setting='off' if debug_mode else 'on') }}" style="display: inline;">
                    <button type="submit" class="debug-toggle">
                        {% if debug_mode %}Disable{% else %}Enable{% endif %} Debug Mode
                    </button>
                </form>
            </div>
        </div>
        
        <div class="section">
            <h2>Recent Reports</h2>
            {% if reports %}
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Report</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {% for report in reports %}
                    <tr>
                        <td>{{ report.date }}</td>
                        <td>{{ report.filename }}</td>
                        <td>
                            <a href="{{ url_for('report', filename=report.filename) }}" target="_blank" class="button">View</a>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p>No reports available. Upload a bank statement to generate reports.</p>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Property Portfolio</h2>
            {% if properties %}
            <table>
                <thead>
                    <tr>
                        <th>Property</th>
                        <th>Address</th>
                        <th>Purchase Date</th>
                        <th>Purchase Price</th>
                    </tr>
                </thead>
                <tbody>
                    {% for property in properties %}
                    <tr>
                        <td>{{ property.name }}</td>
                        <td>{{ property.address }}</td>
                        <td>{{ property.purchase_date }}</td>
                        <td>€{{ property.purchase_price }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p>No properties found in database. Add properties in the database to see them here.</p>
            {% endif %}
        </div>
    </div>
    
    <script>
        // Poll status updates if analysis is in progress
        {% if analysis_status.in_progress %}
        function pollStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('progress-bar').style.width = data.progress + '%';
                    document.getElementById('progress-bar').textContent = data.progress + '%';
                    
                    if (data.error) {
                        document.getElementById('error-message').textContent = data.error;
                        document.getElementById('error-message').classList.remove('hidden');
                    } else {
                        document.getElementById('error-message').classList.add('hidden');
                    }
                    
                    // Check for debug info updates (only if debug mode is on)
                    {% if debug_mode %}
                    if (data.debug_info && data.debug_info.length > 0) {
                        const logContainer = document.querySelector('.debug-log');
                        if (logContainer) {
                            logContainer.innerHTML = '';
                            data.debug_info.forEach(entry => {
                                const div = document.createElement('div');
                                div.className = `debug-entry ${entry.level}`;
                                div.textContent = `[${entry.time}] ${entry.level}: ${entry.message}`;
                                logContainer.appendChild(div);
                            });
                            // Scroll to bottom
                            logContainer.scrollTop = logContainer.scrollHeight;
                        }
                    }
                    {% endif %}
                    
                    if (data.in_progress) {
                        document.getElementById('status-section').classList.remove('hidden');
                        setTimeout(pollStatus, 1000);
                    } else if (!data.error) {
                        // Reload page after successful completion
                        setTimeout(() => {
                            window.location.reload();
                        }, 2000);
                    }
                })
                .catch(error => {
                    console.error('Error polling status:', error);
                    setTimeout(pollStatus, 2000);
                });
        }
        
        // Start polling
        pollStatus();
        {% endif %}
    </script>
</body>
</html>
    ''')

# Start server function
def start_server(port=5000):
    """Start the Flask application server."""
    try:
        # Open browser
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        # Start Flask app
        logger.info(f"Starting Flask app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        print(f"Error starting server: {e}")

if __name__ == '__main__':
    start_server()