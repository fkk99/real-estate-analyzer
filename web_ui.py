"""
Simple web-based UI for Real Estate Investment Analysis
Uses Flask framework - lightweight and easy to use
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import sys
import json
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import webbrowser

# Try to import the analyzer
try:
    from real_estate_analyzer import RealEstateAnalyzer
except ImportError:
    print("Warning: Could not import RealEstateAnalyzer, will use subprocess instead")
    import subprocess

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'csv'}
DB_PATH = 'real_estate_data.db'

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
    'error': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of reports
    reports = []
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
    except Exception as e:
        print(f"Error loading properties: {e}")
    
    # Check if categories are defined
    categories_defined = False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='transactions'")
        if cursor.fetchone()[0] > 0:
            categories_defined = True
        conn.close()
    except:
        pass
    
    return render_template('index.html', 
                          reports=reports, 
                          properties=properties,
                          categories_defined=categories_defined,
                          analysis_status=analysis_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if analysis is already in progress
    if analysis_status['in_progress']:
        flash('Analysis already in progress. Please wait.', 'warning')
        return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get form options
        save_to_db = 'save_to_db' in request.form
        open_report = 'open_report' in request.form
        
        # Start analysis in background thread
        threading.Thread(target=run_analysis, args=(file_path, save_to_db, open_report)).start()
        
        flash(f'File {filename} uploaded and analysis started', 'success')
        return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.', 'error')
    return redirect(url_for('index'))

@app.route('/reports/<filename>')
def report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename)

@app.route('/status')
def status():
    return json.dumps(analysis_status)

def run_analysis(file_path, save_to_db, open_report):
    global analysis_status
    
    analysis_status['in_progress'] = True
    analysis_status['message'] = 'Starting analysis...'
    analysis_status['progress'] = 10
    analysis_status['error'] = None
    
    try:
        # Check if we can import the analyzer
        if 'RealEstateAnalyzer' in globals():
            # Use direct module import
            analyzer = RealEstateAnalyzer(DB_PATH)
            
            # Process CSV file
            analysis_status['message'] = 'Parsing CSV file...'
            analysis_status['progress'] = 20
            df = analyzer.parse_csv(file_path)
            
            if df is not None:
                analysis_status['message'] = 'Categorizing transactions...'
                analysis_status['progress'] = 40
                categorized = analyzer.categorize_transactions()
                
                if categorized is not None:
                    analysis_status['message'] = 'Analyzing data...'
                    analysis_status['progress'] = 60
                    analysis = analyzer.analyze_data()
                    
                    if analysis:
                        # Save to database if option is selected
                        if save_to_db:
                            analysis_status['message'] = 'Saving to database...'
                            analysis_status['progress'] = 80
                            analyzer.save_to_database()
                        
                        # Generate reports
                        analysis_status['message'] = 'Generating reports...'
                        analysis_status['progress'] = 90
                        analyzer.generate_reports(app.config['REPORTS_FOLDER'])
                        
                        report_path = os.path.join(app.config['REPORTS_FOLDER'], 'real_estate_analysis_report.html')
                        
                        # Open report if option is selected
                        if open_report and os.path.exists(report_path):
                            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            
            analyzer.close()
        else:
            # Use subprocess to call the script
            analysis_status['message'] = 'Executing analysis script...'
            analysis_status['progress'] = 30
            
            cmd = [
                sys.executable, 
                "real_estate_analyzer.py", 
                "--input", file_path, 
                "--output", app.config['REPORTS_FOLDER']
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Analysis failed: {stderr}")
            
            analysis_status['progress'] = 90
            
            report_path = os.path.join(app.config['REPORTS_FOLDER'], 'real_estate_analysis_report.html')
            if open_report and os.path.exists(report_path):
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
        
        analysis_status['message'] = 'Analysis complete!'
        analysis_status['progress'] = 100
        
    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['message'] = f"Error: {str(e)}"
    finally:
        analysis_status['in_progress'] = False

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
    </style>
</head>
<body>
    <div class="container">
        <h1>Real Estate Investment Analyzer</h1>
        
        {% for category, message in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ category }}">{{ message }}</div>
        {% endfor %}
        
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
            <div id="error-message" class="flash error {% if not analysis_status.error %}hidden{% endif %}">
                {{ analysis_status.error }}
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
    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    start_server()