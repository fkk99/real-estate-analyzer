"""
Real Estate Investment Analysis System
-------------------------------------
A system to analyze bank statements for real estate investment tracking.
Enhanced with comprehensive logging and debugging features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import re
import traceback
import logging
from datetime import datetime
import locale
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_estate_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealEstateAnalyzer")

# Try to set up locale for Finnish number formatting
try:
    locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')
    logger.info("Set locale to fi_FI.UTF-8")
except Exception as e:
    logger.warning(f"Failed to set Finnish locale: {e}")
    try:
        # Try alternative Finnish locale formats
        locales = ['fi_FI', 'fi', 'Finnish_Finland.1252']
        for loc in locales:
            try:
                locale.setlocale(locale.LC_ALL, loc)
                logger.info(f"Set locale to {loc}")
                break
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Could not set any Finnish locale: {e}")

class RealEstateAnalyzer:
    def __init__(self, db_path="real_estate_data.db"):
        """Initialize the analyzer with database connection."""
        self.db_path = db_path
        self.conn = self._connect_db()
        self.categories = self._load_categories()
        self.current_data = None
        self.analyzed_data = None
        self.debug_mode = False
        logger.info(f"Initialized RealEstateAnalyzer with database: {db_path}")
    
    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode for additional logging."""
        self.debug_mode = enabled
        level = logging.DEBUG if enabled else logging.INFO
        logger.setLevel(level)
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def _safe_str(self, value):
        """Safely convert any value to string."""
        if value is None:
            return ""
        try:
            return str(value)
        except Exception as e:
            logger.warning(f"Error converting {type(value)} to string: {e}")
            return ""
    
    def _safe_float(self, value):
        """Safely convert any value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting {type(value)} value to float: {e}")
            return 0.0
        
    def _connect_db(self):
        """Create or connect to SQLite database."""
        try:
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
            
            logger.info("Database connection established and tables verified")
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _load_categories(self):
        """Load transaction categorization rules."""
        # Default categories with pattern matching rules
        categories = {
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
        logger.info(f"Loaded {len(categories)} transaction categories")
        return categories
    
    def parse_csv(self, file_path):
        """Parse a bank statement CSV file."""
        logger.info(f"Parsing CSV file: {file_path}")
        try:
            # Try different encodings
            encodings = ['utf-8', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    # Try to read with different delimiters
                    for delimiter in [';', ',', '\t']:
                        try:
                            logger.debug(f"Trying delimiter '{delimiter}' with encoding '{encoding}'")
                            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                            # If successful, check if the DataFrame seems valid
                            if len(df.columns) >= 4:  # Expecting at least date, payee, reference, amount
                                logger.info(f"Successfully parsed CSV with encoding: {encoding}, delimiter: {delimiter}")
                                if self.debug_mode:
                                    logger.debug(f"DataFrame head: {df.head().to_dict()}")
                                break
                        except Exception as e:
                            logger.debug(f"Failed with delimiter '{delimiter}': {e}")
                            continue
                    else:
                        # If no delimiter worked, continue to next encoding
                        continue
                    break
                except Exception as e:
                    logger.debug(f"Failed with encoding '{encoding}': {e}")
                    continue
            else:
                # If all encodings failed
                raise ValueError("Could not parse the CSV file with any encoding")
                
            # Try to identify columns based on common names
            logger.debug(f"Original columns: {df.columns.tolist()}")
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
            
            logger.debug(f"Column mapping: {renamed_columns}")
            
            # Rename identified columns
            df = df.rename(columns=renamed_columns)
            
            # Ensure core columns exist
            required_columns = ['date', 'payee', 'type', 'reference', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # Try to infer column names if possible
                for i, col in enumerate(df.columns):
                    if i < len(missing_columns):
                        logger.debug(f"Renaming column '{col}' to '{missing_columns[i]}'")
                        df = df.rename(columns={col: missing_columns[i]})
            
            # Convert amount to numeric, handling different formats
            if 'amount' in df.columns:
                logger.debug("Converting amount column to numeric")
                # Show sample values before conversion for debugging
                if self.debug_mode:
                    logger.debug(f"Amount column before conversion (first 5): {df['amount'].head().tolist()}")
                
                # Replace commas with dots and convert to float
                df['amount'] = df['amount'].astype(str).str.replace(',', '.').astype(float)
                
                if self.debug_mode:
                    logger.debug(f"Amount column after conversion (first 5): {df['amount'].head().tolist()}")
            
            # Extract statement period from filename
            filename = os.path.basename(file_path)
            logger.debug(f"Extracting statement period from filename: {filename}")
            match = re.search(r'(\d{4})_\w+_(\d{4})_\w+', filename)
            if match:
                year1, year2 = match.groups()
                statement_period = f"{year1}-{year2}"
                logger.debug(f"Extracted statement period: {statement_period}")
            else:
                statement_period = datetime.now().strftime("%Y-%m")
                logger.debug(f"Using current date for statement period: {statement_period}")
            
            df['statement_period'] = statement_period
            
            # Store the data
            self.current_data = df
            logger.info(f"Successfully parsed CSV with {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            return None
    
    def categorize_transactions(self):
        """Categorize transactions based on patterns."""
        if self.current_data is None:
            logger.warning("No data loaded to categorize")
            return None
        
        logger.info(f"Categorizing {len(self.current_data)} transactions")
        df = self.current_data.copy()
        df['category'] = 'Other'
        df['confidence'] = 0.5
        
        category_counts = {}
        
        # Apply categorization rules
        for category, rules in self.categories.items():
            # Initialize a mask of all False
            category_mask = pd.Series(False, index=df.index)
            
            # Check each pattern in the rules
            for pattern in rules['patterns']:
                field = pattern.get('field')
                
                if field not in df.columns:
                    logger.warning(f"Field '{field}' not found in data for category '{category}'")
                    continue
                    
                if 'regex' in pattern:
                    # Apply regex pattern
                    logger.debug(f"Applying regex pattern '{pattern['regex']}' to field '{field}' for category '{category}'")
                    field_mask = df[field].astype(str).str.contains(pattern['regex'], na=False)
                    category_mask = category_mask | field_mask
                
                elif 'condition' in pattern:
                    # Apply condition (positive/negative amount)
                    condition = pattern['condition']
                    logger.debug(f"Applying condition '{condition}' to field '{field}' for category '{category}'")
                    if condition == 'positive':
                        field_mask = df['amount'] > 0
                    elif condition == 'negative':
                        field_mask = df['amount'] < 0
                    category_mask = category_mask & field_mask
                
                elif 'threshold' in pattern:
                    # Apply threshold (amount less than threshold)
                    threshold = pattern['threshold']
                    logger.debug(f"Applying threshold {threshold} to field '{field}' for category '{category}'")
                    field_mask = df['amount'] < threshold
                    category_mask = category_mask & field_mask
                
                elif 'abs_threshold' in pattern:
                    # Apply absolute threshold (abs amount greater than threshold)
                    threshold = pattern['abs_threshold']
                    logger.debug(f"Applying absolute threshold {threshold} to field '{field}' for category '{category}'")
                    field_mask = df['amount'].abs() > threshold
                    category_mask = category_mask & field_mask
            
            # Apply category to matches
            matches_count = category_mask.sum()
            if matches_count > 0:
                logger.debug(f"Assigning category '{category}' to {matches_count} transactions")
                df.loc[category_mask, 'category'] = category
                df.loc[category_mask, 'confidence'] = rules['confidence']
                category_counts[category] = matches_count
        
        logger.info(f"Categorization complete. Categories assigned: {category_counts}")
        self.current_data = df
        return df
    
    def analyze_data(self):
        """Generate comprehensive analysis of categorized transactions."""
        if self.current_data is None:
            logger.warning("No data to analyze")
            return None
            
        logger.info("Starting data analysis")
        df = self.current_data
        
        # Extract dates for analysis
        logger.debug("Converting dates for analysis")
        df['date_obj'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        if df['date_obj'].isna().any():
            logger.warning(f"Some dates could not be parsed. Sample problematic values: {df.loc[df['date_obj'].isna(), 'date'].head().tolist()}")
        
        df['month'] = df['date_obj'].dt.month
        df['year'] = df['date_obj'].dt.year
        df['month_year'] = df['date_obj'].dt.strftime('%m.%Y')
        
        # Summary statistics
        logger.debug("Calculating summary statistics")
        deposits_df = df[df['amount'] > 0]
        withdrawals_df = df[df['amount'] < 0]
        
        total_deposits = deposits_df['amount'].sum() if not deposits_df.empty else 0
        total_withdrawals = withdrawals_df['amount'].sum() if not withdrawals_df.empty else 0
        
        logger.debug(f"Total deposits: {total_deposits}, Total withdrawals: {total_withdrawals}")
        
        # Safely get largest deposit and withdrawal
        largest_deposit = {
            "amount": 0,
            "date": "",
            "description": ""
        }
        
        largest_withdrawal = {
            "amount": 0,
            "date": "",
            "description": ""
        }
        
        try:
            if not deposits_df.empty:
                max_idx = deposits_df['amount'].idxmax()
                largest_deposit = {
                    "amount": float(df.loc[max_idx, 'amount']),
                    "date": self._safe_str(df.loc[max_idx, 'date']),
                    "description": self._safe_str(df.loc[max_idx, 'payee']) + " - " + self._safe_str(df.loc[max_idx, 'reference'])
                }
        except Exception as e:
            logger.error(f"Error getting largest deposit: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
        
        try:
            if not withdrawals_df.empty:
                min_idx = withdrawals_df['amount'].idxmin()
                largest_withdrawal = {
                    "amount": float(df.loc[min_idx, 'amount']),
                    "date": self._safe_str(df.loc[min_idx, 'date']),
                    "description": self._safe_str(df.loc[min_idx, 'payee']) + " - " + self._safe_str(df.loc[min_idx, 'reference'])
                }
        except Exception as e:
            logger.error(f"Error getting largest withdrawal: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
        
        summary = {
            "total_transactions": len(df),
            "total_deposits": float(total_deposits),
            "total_withdrawals": float(total_withdrawals),
            "net_change": float(df['amount'].sum()),
            "largest_deposit": largest_deposit,
            "largest_withdrawal": largest_withdrawal
        }
        
        # Monthly analysis
        logger.debug("Performing monthly analysis")
        try:
            monthly = df.groupby('month_year').agg(
                income=('amount', lambda x: float(x[x > 0].sum())),
                expenses=('amount', lambda x: float(x[x < 0].sum())),
                net=('amount', lambda x: float(x.sum())),
                count=('amount', 'count')
            ).reset_index()
            
            # Convert to dictionary with explicit type conversion
            monthly_records = []
            for _, row in monthly.iterrows():
                monthly_records.append({
                    'month_year': self._safe_str(row['month_year']),
                    'income': float(row['income']),
                    'expenses': float(row['expenses']),
                    'net': float(row['net']),
                    'count': int(row['count'])
                })
        except Exception as e:
            logger.error(f"Error in monthly analysis: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            monthly_records = []
        
        # Category analysis
        logger.debug("Performing category analysis")
        try:
            category_summary = df.groupby('category').agg(
                total=('amount', lambda x: float(x.sum())),
                count=('amount', 'count'),
                min=('amount', lambda x: float(x.min())),
                max=('amount', lambda x: float(x.max())),
                mean=('amount', lambda x: float(x.mean()))
            ).reset_index()
            
            # Convert to dictionary with explicit type conversion
            category_records = []
            for _, row in category_summary.iterrows():
                category_records.append({
                    'category': self._safe_str(row['category']),
                    'total': float(row['total']),
                    'count': int(row['count']),
                    'min': float(row['min']),
                    'max': float(row['max']),
                    'mean': float(row['mean'])
                })
        except Exception as e:
            logger.error(f"Error in category analysis: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            category_records = []
        
        # Payee analysis
        logger.debug("Performing payee analysis")
        try:
            payee_summary = df.groupby('payee').agg(
                total=('amount', lambda x: float(x.sum())),
                count=('amount', 'count'),
                min=('amount', lambda x: float(x.min())),
                max=('amount', lambda x: float(x.max())),
                mean=('amount', lambda x: float(x.mean())),
                category=('category', lambda x: self._safe_str(x.mode()[0]) if not x.empty else 'Unknown')
            ).reset_index()
            
            # Convert to dictionary with explicit type conversion
            payee_records = []
            for _, row in payee_summary.iterrows():
                payee_records.append({
                    'payee': self._safe_str(row['payee']),
                    'total': float(row['total']),
                    'count': int(row['count']),
                    'min': float(row['min']),
                    'max': float(row['max']),
                    'mean': float(row['mean']),
                    'category': self._safe_str(row['category'])
                })
        except Exception as e:
            logger.error(f"Error in payee analysis: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            payee_records = []
        
        # Detect recurring transactions
        logger.debug("Identifying recurring transactions")
        try:
            payee_df = pd.DataFrame(payee_records)
            recurring = payee_df[payee_df['count'] > 1].sort_values(by='count', ascending=False)
            recurring_records = recurring.head(10).to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error identifying recurring transactions: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            recurring_records = []
        
        # Look for anomalies
        logger.debug("Detecting anomalies")
        anomalies = []
        
        try:
            # Large transaction anomalies
            deposits_mean = deposits_df['amount'].mean() if not deposits_df.empty else 0
            withdrawals_mean = withdrawals_df['amount'].mean() if not withdrawals_df.empty else 0
            
            logger.debug(f"Deposits mean: {deposits_mean}, Withdrawals mean: {withdrawals_mean}")
            
            # Only look for large deposits if we have enough data
            if not deposits_df.empty and len(deposits_df) > 2:
                large_deposits = df[(df['amount'] > 0) & (df['amount'] > deposits_mean * 3)]
                for _, row in large_deposits.iterrows():
                    anomalies.append({
                        "date": self._safe_str(row['date']),
                        "description": f"{self._safe_str(row['payee'])} - {self._safe_str(row['reference'])}",
                        "amount": float(row['amount']),
                        "reason": "Unusually large deposit",
                        "severity": "High",
                        "confidence": 0.9
                    })
            
            # Only look for large withdrawals if we have enough data
            if not withdrawals_df.empty and len(withdrawals_df) > 2:
                large_withdrawals = df[(df['amount'] < 0) & (df['amount'] < withdrawals_mean * 3)]
                for _, row in large_withdrawals.iterrows():
                    anomalies.append({
                        "date": self._safe_str(row['date']),
                        "description": f"{self._safe_str(row['payee'])} - {self._safe_str(row['reference'])}",
                        "amount": float(row['amount']),
                        "reason": "Unusually large withdrawal",
                        "severity": "High",
                        "confidence": 0.9
                    })
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
        
        logger.debug(f"Found {len(anomalies)} anomalies")
        
        # Cash flow analysis
        logger.debug("Calculating cash flow metrics")
        try:
            monthly_balance = {}
            for record in monthly_records:
                monthly_balance[record['month_year']] = float(record['net'])
        except Exception as e:
            logger.error(f"Error calculating cash flow: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            monthly_balance = {}
        
        # Store the analysis results
        self.analyzed_data = {
            "summary": summary,
            "monthly": monthly_records,
            "categories": category_records,
            "payees": sorted(payee_records, key=lambda x: x['count'], reverse=True)[:10],
            "recurring": recurring_records,
            "anomalies": anomalies,
            "cash_flow": monthly_balance
        }
        
        logger.info("Data analysis complete")
        return self.analyzed_data
    
    def save_to_database(self):
        """Save categorized transactions to database."""
        if self.current_data is None:
            logger.warning("No data to save to database")
            return False
            
        logger.info("Saving data to database")
        try:
            # Save transactions
            transaction_columns = ['date', 'payee', 'type', 'reference', 'amount', 'category', 'statement_period']
            transaction_df = self.current_data[transaction_columns].copy()
            
            # Ensure all columns have appropriate types
            transaction_df['date'] = transaction_df['date'].astype(str)
            transaction_df['payee'] = transaction_df['payee'].astype(str)
            transaction_df['type'] = transaction_df['type'].astype(str)
            transaction_df['reference'] = transaction_df['reference'].astype(str)
            transaction_df['amount'] = transaction_df['amount'].astype(float)
            transaction_df['category'] = transaction_df['category'].astype(str)
            transaction_df['statement_period'] = transaction_df['statement_period'].astype(str)
            
            logger.debug(f"Saving {len(transaction_df)} transactions to database")
            transaction_df.to_sql(
                'transactions', 
                self.conn, 
                if_exists='append', 
                index=False
            )
            
            # Save monthly summary
            if self.analyzed_data:
                logger.debug("Saving monthly summary to database")
                monthly_data = []
                for month_data in self.analyzed_data['monthly']:
                    try:
                        month_year = month_data['month_year'].split('.')
                        monthly_data.append({
                            'month': self._safe_str(month_year[0]),
                            'year': self._safe_str(month_year[1]),
                            'income': self._safe_float(month_data['income']),
                            'expenses': self._safe_float(month_data['expenses']),
                            'net': self._safe_float(month_data['net']),
                            'transaction_count': int(month_data['count'])
                        })
                    except Exception as e:
                        logger.warning(f"Error processing monthly data: {e} - Data: {month_data}")
                
                if monthly_data:
                    monthly_df = pd.DataFrame(monthly_data)
                    monthly_df.to_sql('monthly_summary', self.conn, if_exists='append', index=False)
                else:
                    logger.warning("No valid monthly data to save")
            
            logger.info("Database save complete")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            return False
    def _safe_str(self, value):
        """Safely convert any value to string."""
        if value is None:
            return ""
        try:
            return str(value)
        except Exception as e:
            logger.warning(f"Error converting {type(value)} to string: {e}")
            return ""

    def _safe_float(self, value):
        """Safely convert any value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting {type(value)} value to float: {e}")
            return 0.0
    
    def generate_reports(self, output_dir="."):
        """Generate reports and visualizations from analyzed data."""
        if self.analyzed_data is None:
            logger.warning("No analyzed data for report generation")
            return False
            
        logger.info(f"Generating reports in directory: {output_dir}")
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        try:
            logger.debug("Generating JSON report")
            json_path = os.path.join(output_dir, 'real_estate_analysis_report.json')
            with open(json_path, 'w') as f:
                json.dump(self.analyzed_data, f, indent=4)
            logger.debug(f"JSON report saved to {json_path}")
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
        
        # Generate plots
        try:
            if self.analyzed_data['monthly']:
                logger.debug("Generating visualization plots")
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
                monthly_plot_path = os.path.join(output_dir, 'monthly_analysis.png')
                plt.savefig(monthly_plot_path, dpi=300)
                logger.debug(f"Monthly analysis plot saved to {monthly_plot_path}")
                
                # Additional chart: Top payees
                plt.figure(figsize=(12, 6))
                payee_df = pd.DataFrame(self.analyzed_data['payees']).head(10)
                payee_df = payee_df.sort_values('total')
                payee_df.plot(x='payee', y='total', kind='barh', ax=plt.gca())
                plt.title('Top 10 Transaction Partners by Volume')
                plt.tight_layout()
                payees_plot_path = os.path.join(output_dir, 'top_payees.png')
                plt.savefig(payees_plot_path, dpi=300)
                logger.debug(f"Top payees plot saved to {payees_plot_path}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
        
        # Generate HTML report
        try:
            logger.debug("Generating HTML report")
            html_report = self._generate_html_report()
            html_path = os.path.join(output_dir, 'real_estate_analysis_report.html')
            with open(html_path, 'w') as f:
                f.write(html_report)
            logger.debug(f"HTML report saved to {html_path}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            
        logger.info(f"Report generation complete in {output_dir}")
        return True
    def _generate_html_report(self):
        """Generate HTML report from analyzed data."""
        logger.debug("Starting HTML report generation")
        
        if not self.analyzed_data:
            logger.warning("No analyzed data for HTML report")
            return "<html><body><h1>No data to display</h1></body></html>"
            
        try:
            data = self.analyzed_data
            
            # Debug info for troubleshooting
            if self.debug_mode:
                logger.debug("Data keys available for HTML report: " + str(list(data.keys())))
                logger.debug("Monthly data sample: " + str(data['monthly'][0] if data['monthly'] else "None"))
                logger.debug("Categories data sample: " + str(data['categories'][0] if data['categories'] else "None"))
                logger.debug("Recurring data sample: " + str(data['recurring'][0] if data['recurring'] else "None"))
            
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
                                <td>{self._safe_str(data['summary']['total_transactions'])}</td>
                            </tr>
                            <tr>
                                <td>Total Deposits</td>
                                <td class="positive">€{self._safe_float(data['summary']['total_deposits']):.2f}</td>
                            </tr>
                            <tr>
                                <td>Total Withdrawals</td>
                                <td class="negative">€{self._safe_float(data['summary']['total_withdrawals']):.2f}</td>
                            </tr>
                            <tr>
                                <td>Net Change</td>
                                <td class="{'positive' if self._safe_float(data['summary']['net_change']) >= 0 else 'negative'}">
                                    €{self._safe_float(data['summary']['net_change']):.2f}
                                </td>
                            </tr>
                            <tr>
                                <td>Largest Deposit</td>
                                <td>
                                    €{self._safe_float(data['summary']['largest_deposit']['amount']):.2f} on 
                                    {self._safe_str(data['summary']['largest_deposit']['date'])} - 
                                    {self._safe_str(data['summary']['largest_deposit']['description'])}
                                </td>
                            </tr>
                            <tr>
                                <td>Largest Withdrawal</td>
                                <td>
                                    €{self._safe_float(data['summary']['largest_withdrawal']['amount']):.2f} on 
                                    {self._safe_str(data['summary']['largest_withdrawal']['date'])} - 
                                    {self._safe_str(data['summary']['largest_withdrawal']['description'])}
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
            logger.debug(f"Adding {len(data['monthly'])} monthly rows to HTML report")
            for month in data['monthly']:
                try:
                    month_year = self._safe_str(month['month_year'])
                    income = self._safe_float(month['income'])
                    expenses = self._safe_float(month['expenses'])
                    net = self._safe_float(month['net'])
                    count = int(month['count']) if 'count' in month else 0
                    
                    html += f"""
                        <tr>
                            <td>{month_year}</td>
                            <td class="positive">€{income:.2f}</td>
                            <td class="negative">€{expenses:.2f}</td>
                            <td class="{'positive' if net >= 0 else 'negative'}">€{net:.2f}</td>
                            <td>{count}</td>
                        </tr>
                    """
                except Exception as e:
                    logger.warning(f"Error adding monthly row: {e} - Data: {month}")
                    continue
                
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
            logger.debug(f"Adding {len(data['categories'])} category rows to HTML report")
            for category in data['categories']:
                try:
                    category_name = self._safe_str(category['category'])
                    total = self._safe_float(category['total'])
                    count = int(category['count']) if 'count' in category else 0
                    mean = self._safe_float(category['mean'])
                    
                    html += f"""
                        <tr>
                            <td>{category_name}</td>
                            <td class="{'positive' if total >= 0 else 'negative'}">€{total:.2f}</td>
                            <td>{count}</td>
                            <td>€{mean:.2f}</td>
                        </tr>
                    """
                except Exception as e:
                    logger.warning(f"Error adding category row: {e} - Data: {category}")
                    continue
                
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
            logger.debug(f"Adding {len(data['recurring'])} recurring relationship rows to HTML report")
            for payee in data['recurring']:
                try:
                    payee_name = self._safe_str(payee['payee'])
                    total = self._safe_float(payee['total'])
                    count = int(payee['count']) if 'count' in payee else 0
                    mean = self._safe_float(payee['mean'])
                    category = self._safe_str(payee['category'])
                    
                    html += f"""
                        <tr>
                            <td>{payee_name}</td>
                            <td class="{'positive' if total >= 0 else 'negative'}">€{total:.2f}</td>
                            <td>{count}</td>
                            <td>€{mean:.2f}</td>
                            <td>{category}</td>
                        </tr>
                    """
                except Exception as e:
                    logger.warning(f"Error adding recurring row: {e} - Data: {payee}")
                    continue
                
            html += """
                        </table>
                    </div>
            """
            
            # Add anomalies section if there are any
            if data['anomalies']:
                logger.debug(f"Adding {len(data['anomalies'])} anomalies to HTML report")
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
                    try:
                        date = self._safe_str(anomaly['date'])
                        description = self._safe_str(anomaly['description'])
                        amount = self._safe_float(anomaly['amount'])
                        reason = self._safe_str(anomaly['reason'])
                        severity = self._safe_str(anomaly['severity'])
                        
                        html += f"""
                            <tr>
                                <td>{date}</td>
                                <td>{description}</td>
                                <td class="{'positive' if amount >= 0 else 'negative'}">€{amount:.2f}</td>
                                <td>{reason}</td>
                                <td>{severity}</td>
                            </tr>
                        """
                    except Exception as e:
                        logger.warning(f"Error adding anomaly row: {e} - Data: {anomaly}")
                        continue
                    
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
            
            logger.debug("HTML report generation complete")
            return html
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            if self.debug_mode:
                logger.error(traceback.format_exc())
            return "<html><body><h1>Error generating report</h1><p>Check logs for details</p></body></html>"

    def close(self):
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")