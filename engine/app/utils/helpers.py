import os
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import re
import logging
from decimal import Decimal

from app.config import settings

logger = logging.getLogger(__name__)

def generate_unique_id() -> str:
    """Generate a unique ID for uploaded files"""
    return str(uuid.uuid4())

def normalize_invoice_number(invoice_num: Any) -> str:
    """
    Normalize invoice number for consistent matching
    
    Args:
        invoice_num: Invoice number in any format
        
    Returns:
        Normalized invoice number string
    """
    if pd.isna(invoice_num):
        return ""
    
    # Convert to string first
    invoice_str = str(invoice_num).strip()
    
    # Remove common separators and whitespace
    invoice_str = re.sub(r'[\s/\-_.]', '', invoice_str)
    
    # Convert to uppercase
    invoice_str = invoice_str.upper()
    
    return invoice_str

def normalize_date(date_value: Any, input_format: str = '%d-%b-%Y') -> str:
    """
    Normalize date string to standard format (DD-MM-YYYY)
    
    Args:
        date_value: Date in various formats
        input_format: Expected input format string
    
    Returns:
        Normalized date string in DD-MM-YYYY format
    """
    if pd.isna(date_value):
        return ""
    
    try:
        # If already a datetime object
        if isinstance(date_value, datetime):
            return date_value.strftime('%d-%m-%Y')
        
        # Try parsing as string
        date_str = str(date_value).strip()
        
        # Try with the specified format
        try:
            parsed_date = datetime.strptime(date_str, input_format)
            return parsed_date.strftime('%d-%m-%Y')
        except ValueError:
            # Try with other common formats
            for fmt in ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%Y', '%d.%m.%Y']:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%d-%m-%Y')
                except ValueError:
                    continue
            
            # If all formats fail, try pandas to_datetime
            try:
                parsed_date = pd.to_datetime(date_str)
                return parsed_date.strftime('%d-%m-%Y')
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return date_str
                
    except Exception as e:
        logger.error(f"Error normalizing date {date_value}: {e}")
        return str(date_value)

def normalize_decimal(value: Any) -> float:
    """
    Normalize numeric values for consistent comparison
    
    Args:
        value: Numeric value in any format
        
    Returns:
        Normalized float value
    """
    if pd.isna(value):
        return 0.0
    
    try:
        if isinstance(value, str):
            # Remove commas and other formatting
            clean_value = re.sub(r'[^\d.-]', '', value)
            return float(clean_value) if clean_value else 0.0
        elif isinstance(value, (int, float, Decimal)):
            return float(value)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error normalizing decimal {value}: {e}")
        return 0.0

def save_json_file(data: Dict[str, Any], filename: str) -> str:
    """
    Save data to a JSON file
    
    Args:
        data: Dictionary to save
        filename: Filename to save to
        
    Returns:
        Path to the saved file
    """
    file_path = os.path.join(settings.REPORTS_DIR, filename)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")
        raise
        
def json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename"""
    return os.path.splitext(filename)[1].lower()

def is_valid_gstr2b_json(data: Dict[str, Any]) -> bool:
    """
    Validate if the JSON data follows GSTR-2B format
    
    Args:
        data: JSON data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check for required keys in the structure
        if not isinstance(data, dict):
            return False
            
        if 'data' not in data or not isinstance(data['data'], dict):
            return False
            
        gstr_data = data['data']
        required_keys = ['docdata', 'gstin', 'rtnprd']
        
        for key in required_keys:
            if key not in gstr_data:
                return False
        
        # Check for B2B invoices structure
        if 'docdata' in gstr_data and isinstance(gstr_data['docdata'], dict):
            if 'b2b' not in gstr_data['docdata'] or not isinstance(gstr_data['docdata']['b2b'], list):
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating GSTR-2B JSON: {e}")
        return False

def validate_purchase_csv(df: pd.DataFrame) -> bool:
    """
    Validate if the CSV data has required columns for purchase register
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = [
        'GSTIN of Supplier', 'Invoice Number', 'Invoice date', 
        'Invoice Value', 'Taxable Value'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns in purchase CSV: {missing_columns}")
        return False
        
    return True

def identify_gstin_from_gstr2b(data: Dict[str, Any]) -> Optional[str]:
    """Extract the GSTIN from GSTR-2B data"""
    try:
        return data.get('data', {}).get('gstin')
    except Exception:
        return None

def identify_return_period(data: Dict[str, Any]) -> Optional[str]:
    """Extract the return period from GSTR-2B data"""
    try:
        return data.get('data', {}).get('rtnprd')
    except Exception:
        return None