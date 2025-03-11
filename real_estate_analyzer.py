"""
Real Estate Investment Analysis System
-------------------------------------
A system to analyze bank statements for real estate investment tracking.
- Parses CSV bank statements
- Categorizes transactions
- Calculates key metrics
- Generates reports and visualizations
- Maintains historical data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import re
from datetime import datetime
import locale
from pathlib import Path
import argparse

# Set up locale for Finnish number formatting
locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')

class RealEstateAnalyzer:
    def __init__(self, db_path="real_estate_data.db"):
        """Initialize the analyzer with database connection."""
        self.db_path = db_path
        self.conn = self._connect_db()
        self.categories = self._load_categories()
        self.current_data = None
        self.analyzed_data = None
    
    def _connect_db(self):
        """Create or connect to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables if they don't exist
        conn.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            payee TEXT,
            type TEXT,
            reference TEXT,
            amount REAL,
            category TEXT,
            statement_period TEXT
        )
        ''')
        
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
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS monthly_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT,
            year TEXT, 
            income REAL,
            expenses REAL,
            net REAL,
            transaction_count INTEGER
        )
        ''')
        
        return conn
    
    def _load_categories(self):
        """Load transaction categorization rules."""
        # Default categories with pattern matching rules
        return {
            "rental_income": {
                "patterns": [
                    {"field": "reference", "regex": r"(?i)vuokra|rent|luhtikatu"},
                    {"field": "amount", "condition": "positive"}
                ],
                "confidence": 0.95
            },
            "housing_association_fees": {
                "patterns": [
                    {"field": "payee", "regex": r"(?i)asunto-oy|as\. oy"},
                    {"field": "amount", "condition": "negative"}
                ],
                "confidence": 0.90
            },
            "loan_payment": {
                "patterns": [
                    {"field": "type", "regex": r"(?i)laina|loan|lainanhoito|lainat"},
                    {"field": "amount", "condition": "negative"}
                ],
                "confidence": 0.90
            },
            "loan_disbursement": {
                "patterns": [
                    {"field": "type", "regex": r"(?i)laina|loan|lainanhoito|lainat"},
                    {"field": "amount", "condition": "positive"}
                ],
                "confidence": 0.85
            },
            "membership_fee": {
                "patterns": [
                    {"field": "reference", "regex": r"(?i)jäsenmaksu|senmaksu"}
                ],
                "confidence": 0.95
            },
            "bank_fees": {
                "patterns": [
                    {"field": "type", "regex": r"(?i)palvelumaksu"},
                ],
                "confidence": 0.95
            },
            "property_purchase": {
                "patterns": [
                    {"field": "amount", "threshold": -5000},
                    {"field": "reference", "regex": r"(?i)kauppahinta|purchase|asunto"}
                ],
                "confidence": 0.85
            },
            "large_transfer": {
                "patterns": [
                    {"field": "amount", "abs_threshold": 5000}
                ],
                "confidence": 0.80
            }
        }
    
    def parse_csv(self, file_path):
        """Parse a bank statement CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Try to read with different delimiters
                    for delimiter in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                            # If successful, check if the DataFrame seems valid
                            if len(df.columns) >= 4:  # Expecting at least date, payee, reference, amount
                                print(f"Successfully parsed CSV with encoding: {encoding}, delimiter: {delimiter}")
                                break
                        except:
                            continue
                    else:
                        # If no delimiter worked, continue to next encoding
                        continue
                    break
                except:
                    continue
            else:
                # If all encodings failed
                raise ValueError("Could not parse the CSV file with any encoding")
                
            # Try to identify columns based on common names
            renamed_columns = {}
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['päivä', 'pvm', 'date']):
                    renamed_columns[col] = 'date'
                elif any(x in col_lower for x in ['saaja', 'maksaja', 'payee', 'payer']):
                    renamed_columns[col] = 'payee'
                elif any(x in col_lower for x in ['selite', 'type', 'desc']):
                    renamed_columns[col] = 'type'
                elif any(x in col_lower for x in ['viite', 'viesti', 'message', 'reference']):
                    renamed_columns[col] = 'reference'
                elif any(x in col_lower for x in ['määrä', 'summa', 'amount']):
                    renamed_columns[col] = 'amount'
            
            # Rename identified columns
            df = df.rename(columns=renamed_columns)
            
            # Ensure core columns exist
            required_columns = ['date', 'payee', 'type', 'reference', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
                # Try to infer column names if possible
                for i, col in enumerate(df.columns):
                    if i < len(missing_columns):
                        df = df.rename(columns={col: missing_columns[i]})
            
            # Convert amount to numeric, handling different formats
            if 'amount' in df.columns:
                df['amount'] = df['amount'].astype(str).str.replace(',', '.').astype(float)
            
            # Extract statement period from filename
            filename = os.path.basename(file_path)
            match = re.search(r'(\d{4})_\w+_(\d{4})_\w+', filename)
            if match:
                year1, year2 = match.groups()
                statement_period = f"{year1}-{year2}"
            else:
                statement_period = datetime.now().strftime("%Y-%m")
            
            df['statement_period'] = statement_period
            
            # Store the data
            self.current_data = df
            
            return df
            
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return None
    
    def categorize_transactions(self):
        """Categorize transactions based on patterns."""
        if self.current_data is None:
            print("No data loaded to categorize")
            return None
        
        df = self.current_data.copy()
        df['category'] = 'Other'
        df['confidence'] = 0.5
        
        # Apply categorization rules
        for category, rules in self.categories.items():
            # Initialize a mask of all False
            category_mask = pd.Series(False, index=df.index)
            
            # Check each pattern in the rules
            for pattern in rules['patterns']:
                field = pattern.get('field')
                
                if field not in df.columns:
                    continue
                    
                if 'regex' in pattern:
                    # Apply regex pattern
                    field_mask = df[field].astype(str).str.contains(pattern['regex'], na=False)
                    category_mask = category_mask | field_mask
                
                elif 'condition' in pattern:
                    # Apply condition (positive/negative amount)
                    if pattern['condition'] == 'positive':
                        field_mask = df['amount'] > 0
                    elif pattern['condition'] == 'negative':
                        field_mask = df['amount'] < 0
                    category_mask = category_mask & field_mask
                
                elif 'threshold' in pattern:
                    # Apply threshold (amount less than threshold)
                    threshold = pattern['threshold']
                    field_mask = df['amount'] < threshold
                    category_mask = category_mask & field_mask
                
                elif 'abs_threshold' in pattern:
                    # Apply absolute threshold (abs amount greater than threshold)
                    threshold = pattern['abs_threshold']
                    field_mask = df['amount'].abs() > threshold
                    category_mask = category_mask & field_mask
            
            # Apply category to matches
            df.loc[category_mask, 'category'] = category
            df.loc[category_mask, 'confidence'] = rules['confidence']
        
        self.current_data = df
        return df
    
    def analyze_data(self):
        """Generate comprehensive analysis of categorized transactions."""
        if self.current_data is None:
            print("No data to analyze")
            return None
            
        df = self.current_data
        
        # Extract dates for analysis
        df['date_obj'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        df['month'] = df['date_obj'].dt.month
        df['year'] = df['date_obj'].dt.year
        df['month_year'] = df['date_obj'].dt.strftime('%m.%Y')
        
        # Summary statistics
        summary = {
            "total_transactions": len(df),
            "total_deposits": df[df['amount'] > 0]['amount'].sum(),
            "total_withdrawals": df[df['amount'] < 0]['amount'].sum(),
            "net_change": df['amount'].sum(),
            "largest_deposit": {
                "amount": df[df['amount'] > 0]['amount'].max() if not df[df['amount'] > 0].empty else 0,
                "date": df.loc[df['amount'].idxmax(), 'date'] if not df.empty else None,
                "description": df.loc[df['amount'].idxmax(), 'payee'] + " - " + df.loc[df['amount'].idxmax(), 'reference'] if not df.empty else None
            },
            "largest_withdrawal": {
                "amount": df[df['amount'] < 0]['amount'].min() if not df[df['amount'] < 0].empty else 0,
                "date": df.loc[df['amount'].idxmin(), 'date'] if not df.empty else None,
                "description": df.loc[df['amount'].idxmin(), 'payee'] + " - " + df.loc[df['amount'].idxmin(), 'reference'] if not df.empty else None
            }
        }
        
        # Monthly analysis
        monthly = df.groupby('month_year').agg(
            income=('amount', lambda x: x[x > 0].sum()),
            expenses=('amount', lambda x: x[x < 0].sum()),
            net=('amount', 'sum'),
            count=('amount', 'count')
        ).reset_index()
        
        # Category analysis
        category_summary = df.groupby('category').agg(
            total=('amount', 'sum'),
            count=('amount', 'count'),
            min=('amount', 'min'),
            max=('amount', 'max'),
            mean=('amount', 'mean')
        ).reset_index()
        
        # Payee analysis
        payee_summary = df.groupby('payee').agg(
            total=('amount', 'sum'),
            count=('amount', 'count'),
            min=('amount', 'min'),
            max=('amount', 'max'),
            mean=('amount', 'mean'),
            category=('category', lambda x: x.mode()[0] if not x.empty else 'Unknown')
        ).reset_index()
        
        # Detect recurring transactions
        recurring = payee_summary[payee_summary['count'] > 1].sort_values(by='count', ascending=False)
        
        # Look for anomalies
        anomalies = []
        
        # Large transaction anomalies
        large_deposits = df[(df['amount'] > 0) & (df['amount'] > df[df['amount'] > 0]['amount'].mean() * 3)]
        large_withdrawals = df[(df['amount'] < 0) & (df['amount'] < df[df['amount'] < 0]['amount'].mean() * 3)]
        
        for _, row in large_deposits.iterrows():
            anomalies.append({
                "date": row['date'],
                "description": f"{row['payee']} - {row['reference']}",
                "amount": row['amount'],
                "reason": "Unusually large deposit",
                "severity": "High",
                "confidence": 0.9
            })
            
        for _, row in large_withdrawals.iterrows():
            anomalies.append({
                "date": row['date'],
                "description": f"{row['payee']} - {row['reference']}",
                "amount": row['amount'],
                "reason": "Unusually large withdrawal",
                "severity": "High",
                "confidence": 0.9
            })
        
        # Cash flow analysis
        monthly_balance = monthly.set_index('month_year')['net'].to_dict()
        
        # Store the analysis results
        self.analyzed_data = {
            "summary": summary,
            "monthly": monthly.to_dict(orient='records'),
            "categories": category_summary.to_dict(orient='records'),
            "payees": payee_summary.sort_values(by='count', ascending=False).head(10).to_dict(orient='records'),
            "recurring": recurring.head(10).to_dict(orient='records'),
            "anomalies": anomalies,
            "cash_flow": monthly_balance
        }
        
        return self.analyzed_data
    
    def save_to_database(self):
        """Save categorized transactions to database."""
        if self.current_data is None:
            print("No data to save")
            return False
            
        try:
            # Save transactions
            self.current_data[['date', 'payee', 'type', 'reference', 'amount', 'category', 'statement_period']].to_sql(
                'transactions', 
                self.conn, 
                if_exists='append', 
                index=False
            )
            
            # Save monthly summary
            if self.analyzed_data:
                monthly_data = []
                for month_data in self.analyzed_data['monthly']:
                    month_year = month_data['month_year'].split('.')
                    monthly_data.append({
                        'month': month_year[0],
                        'year': month_year[1],
                        'income': month_data['income'],
                        'expenses': month_data['expenses'],
                        'net': month_data['net'],
                        'transaction_count': month_data['count']
                    })
                
                monthly_df = pd.DataFrame(monthly_data)
                monthly_df.to_sql('monthly_summary', self.conn, if_exists='append', index=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def generate_reports(self, output_dir="."):
        """Generate reports and visualizations from analyzed data."""
        if self.analyzed_data is None:
            print("No analyzed data to report on")
            return False
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        with open(os.path.join(output_dir, 'real_estate_analysis_report.json'), 'w') as f:
            json.dump(self.analyzed_data, f, indent=4)
        
        # Generate plots
        if self.analyzed_data['monthly']:
            plt.figure(figsize=(12, 6))
            
            # Convert to DataFrame for easier plotting
            monthly_df = pd.DataFrame(self.analyzed_data['monthly'])
            
            # Monthly income/expenses bar chart
            plt.subplot(2, 2, 1)
            monthly_df.plot(x='month_year', y=['income', 'expenses'], kind='bar', ax=plt.gca())
            plt.title('Monthly Income vs Expenses')
            plt.tight_layout()
            
            # Net cash flow line chart
            plt.subplot(2, 2, 2)
            monthly_df.plot(x='month_year', y='net', kind='line', marker='o', ax=plt.gca())
            plt.title('Monthly Net Cash Flow')
            plt.tight_layout()
            
            # Category breakdown pie chart
            plt.subplot(2, 2, 3)
            category_df = pd.DataFrame(self.analyzed_data['categories'])
            income_categories = category_df[category_df['total'] > 0]
            if not income_categories.empty:
                income_categories.plot.pie(y='total', labels=income_categories['category'], autopct='%1.1f%%', ax=plt.gca())
                plt.title('Income by Category')
                plt.ylabel('')
            
            # Expense breakdown pie chart
            plt.subplot(2, 2, 4)
            expense_categories = category_df[category_df['total'] < 0].copy()
            if not expense_categories.empty:
                expense_categories['total_abs'] = expense_categories['total'].abs()
                expense_categories.plot.pie(y='total_abs', labels=expense_categories['category'], autopct='%1.1f%%', ax=plt.gca())
                plt.title('Expenses by Category')
                plt.ylabel('')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'monthly_analysis.png'), dpi=300)
            
            # Additional chart: Top payees
            plt.figure(figsize=(12, 6))
            payee_df = pd.DataFrame(self.analyzed_data['payees']).head(10)
            payee_df = payee_df.sort_values('total')
            payee_df.plot(x='payee', y='total', kind='barh', ax=plt.gca())
            plt.title('Top 10 Transaction Partners by Volume')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_payees.png'), dpi=300)
        
        # Generate HTML report
        html_report = self._generate_html_report()
        with open(os.path.join(output_dir, 'real_estate_analysis_report.html'), 'w') as f:
            f.write(html_report)
            
        print(f"Reports generated in {output_dir}")
        return True
    
    def _generate_html_report(self):
        """Generate HTML report from analyzed data."""
        if not self.analyzed_data:
            return "<html><body><h1>No data to display</h1></body></html>"
            
        data = self.analyzed_data
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real Estate Investment Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ width: 48%; margin-bottom: 20px; }}
                @media (max-width: 800px) {{ .chart {{ width: 100%; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Real Estate Investment Analysis Report</h1>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <div class="section">
                    <h2>Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Transactions</td>
                            <td>{data['summary']['total_transactions']}</td>
                        </tr>
                        <tr>
                            <td>Total Deposits</td>
                            <td class="positive">€{data['summary']['total_deposits']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Total Withdrawals</td>
                            <td class="negative">€{data['summary']['total_withdrawals']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Net Change</td>
                            <td class="{'positive' if data['summary']['net_change'] >= 0 else 'negative'}">
                                €{data['summary']['net_change']:.2f}
                            </td>
                        </tr>
                        <tr>
                            <td>Largest Deposit</td>
                            <td>
                                €{data['summary']['largest_deposit']['amount']:.2f} on 
                                {data['summary']['largest_deposit']['date']} - 
                                {data['summary']['largest_deposit']['description']}
                            </td>
                        </tr>
                        <tr>
                            <td>Largest Withdrawal</td>
                            <td>
                                €{data['summary']['largest_withdrawal']['amount']:.2f} on 
                                {data['summary']['largest_withdrawal']['date']} - 
                                {data['summary']['largest_withdrawal']['description']}
                            </td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Monthly Analysis</h2>
                    <table>
                        <tr>
                            <th>Month</th>
                            <th>Income</th>
                            <th>Expenses</th>
                            <th>Net</th>
                            <th>Transactions</th>
                        </tr>
        """
        
        # Add monthly rows
        for month in data['monthly']:
            html += f"""
                <tr>
                    <td>{month['month_year']}</td>
                    <td class="positive">€{month['income']:.2f}</td>
                    <td class="negative">€{month['expenses']:.2f}</td>
                    <td class="{'positive' if month['net'] >= 0 else 'negative'}">€{month['net']:.2f}</td>
                    <td>{month['count']}</td>
                </tr>
            """
            
        html += """
                    </table>
                </div>
                
                <div class="chart-container">
                    <div class="chart">
                        <img src="monthly_analysis.png" alt="Monthly Analysis" style="width: 100%;">
                    </div>
                    <div class="chart">
                        <img src="top_payees.png" alt="Top Payees" style="width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Category Analysis</h2>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Total</th>
                            <th>Count</th>
                            <th>Average</th>
                        </tr>
        """
        
        # Add category rows
        for category in data['categories']:
            html += f"""
                <tr>
                    <td>{category['category']}</td>
                    <td class="{'positive' if category['total'] >= 0 else 'negative'}">€{category['total']:.2f}</td>
                    <td>{category['count']}</td>
                    <td>€{category['mean']:.2f}</td>
                </tr>
            """
            
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Top Recurring Relationships</h2>
                    <table>
                        <tr>
                            <th>Payee</th>
                            <th>Total</th>
                            <th>Count</th>
                            <th>Average</th>
                            <th>Category</th>
                        </tr>
        """
        
        # Add recurring relationship rows
        for payee in data['recurring']:
            html += f"""
                        <tr>
                            <td>{payee['payee']}</td>
                            <td class="{'positive' if float(payee['total']) >= 0 else 'negative'}">€{float(payee['total']):.2f}</td>
                            <td>{payee['count']}</td>
                            <td>€{float(payee['mean']):.2f}</td>
                            <td>{str(payee['category'])}</td>
                        </tr>
        """
            
        html += """
                    </table>
                </div>
        """
        
        # Add anomalies section if there are any
        if data['anomalies']:
            html += """
                <div class="section">
                    <h2>Detected Anomalies</h2>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Description</th>
                            <th>Amount</th>
                            <th>Reason</th>
                            <th>Severity</th>
                        </tr>
            """
            
            for anomaly in data['anomalies']:
                html += f"""
                    <tr>
                        <td>{anomaly['date']}</td>
                        <td>{anomaly['description']}</td>
                        <td class="{'positive' if anomaly['amount'] >= 0 else 'negative'}">€{anomaly['amount']:.2f}</td>
                        <td>{anomaly['reason']}</td>
                        <td>{anomaly['severity']}</td>
                    </tr>
                """
                
            html += """
                    </table>
                </div>
            """
            
        # Close the HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main function to run the analyzer from command line."""
    parser = argparse.ArgumentParser(description='Real Estate Investment Analysis System')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='./output', help='Output directory for reports')
    args = parser.parse_args()
    
    print(f"Analyzing {args.input}...")
    analyzer = RealEstateAnalyzer()
    
    # Process CSV file
    df = analyzer.parse_csv(args.input)
    if df is not None:
        print(f"Loaded {len(df)} transactions.")
        
        # Categorize transactions
        categorized = analyzer.categorize_transactions()
        if categorized is not None:
            print(f"Categorized {len(categorized)} transactions.")
            
            # Analyze data
            analysis = analyzer.analyze_data()
            if analysis:
                print("Analysis complete.")
                
                # Save to database
                if analyzer.save_to_database():
                    print("Data saved to database.")
                
                # Generate reports
                if analyzer.generate_reports(args.output):
                    print(f"Reports generated in {args.output}")
    
    analyzer.close()
    print("Analysis complete.")

if __name__ == "__main__":
    main()