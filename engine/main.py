from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid
import io
import json
import logging
import os
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from fuzzywuzzy import fuzz
import asyncio
import redis
from sentence_transformers import SentenceTransformer
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gst-reconciliation-ml")

# Create FastAPI app
app = FastAPI(
    title="GST Reconciliation API with ML",
    description="API for reconciling GSTR-2B data with Purchase Register data using ML",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables with defaults
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RETAIN_JOBS_HOURS = int(os.getenv("RETAIN_JOBS_HOURS", "24"))
USE_ML = os.getenv("USE_ML", "true").lower() == "true"  # Enable/disable ML features

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Initialize Redis client for job storage
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  # Test connection
    use_redis = True
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Falling back to in-memory storage.")
    use_redis = False
    job_statuses = {}  # Fallback in-memory storage

# Global ML model cache
ml_models = {
    "matching_model": None,
    "anomaly_model": None,
    "embedding_model": None,
}

# Pydantic models (expanded with ML parameters)
class ReconciliationRequest(BaseModel):
    # Traditional parameters
    tolerance_amount: float = 1.0
    tolerance_percentage: float = 2.0
    fuzzy_match_threshold: int = 85
    date_exact_tolerance: int = 1
    date_fuzzy_tolerance: int = 7
    
    # ML parameters
    use_ml: bool = True
    ml_confidence_threshold: float = 0.7
    detect_anomalies: bool = True
    anomaly_sensitivity: float = 0.1

class InvoiceDetail(BaseModel):
    gstin: str
    invoice_number: str
    invoice_date: str
    taxable_value: float
    igst: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    total_tax: float = 0.0
    total_amount: float = 0.0
    match_score: Optional[float] = None
    date_delta: Optional[int] = None
    amount_difference: Optional[float] = None
    ml_confidence: Optional[float] = None  # Added ML confidence score
    is_anomaly: Optional[bool] = None      # Flag for anomalous invoices

class MismatchedInvoice(InvoiceDetail):
    reason: str
    gstr2b_details: Dict[str, Any]
    pr_details: Dict[str, Any]
    correction_suggestion: Optional[str] = None  # Added ML-based suggestion

class ReconciliationSummary(BaseModel):
    total_gstr2b_invoices: int
    total_purchase_invoices: int
    matched_count: int
    mismatched_count: int
    missing_in_gstr2b: int
    missing_in_purchase_register: int
    anomaly_count: int = 0  # Added anomaly count

class ReconciliationResponse(BaseModel):
    matched: List[InvoiceDetail]
    mismatched: List[MismatchedInvoice]
    only_in_gstr2b: List[InvoiceDetail]
    only_in_purchase_register: List[InvoiceDetail]
    summary: ReconciliationSummary
    potential_matches: Optional[List[Dict[str, Any]]] = None  # ML-suggested potential matches

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

class MLTrainingRequest(BaseModel):
    use_previous_data: bool = True
    training_files: Optional[List[str]] = None
    retrain_matching_model: bool = True
    retrain_anomaly_model: bool = True

class MLTrainingStatus(BaseModel):
    job_id: str
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    job_id: str
    invoice_id: str  # Either GSTR-2B or PR identifier
    feedback_type: str  # "correct_match", "incorrect_match", "correct_suggestion", etc.
    correct_match_id: Optional[str] = None  # If providing correct match
    notes: Optional[str] = None

# Helper functions for job status management
def set_job_status(job_id, status_data):
    if use_redis:
        # Store with expiration to prevent memory leaks
        expire_seconds = RETAIN_JOBS_HOURS * 3600
        status_data["timestamp"] = datetime.now().isoformat()
        redis_client.set(f"job:{job_id}", json.dumps(status_data), ex=expire_seconds)
    else:
        # Use in-memory storage
        job_statuses[job_id] = {**status_data, "timestamp": datetime.now()}

def get_job_status(job_id):
    if use_redis:
        data = redis_client.get(f"job:{job_id}")
        return json.loads(data) if data else None
    else:
        return job_statuses.get(job_id)

def store_reconciliation_data(job_id, gstr2b_df, pr_df, result):
    """Store reconciliation data for future ML training"""
    try:
        # Create a directory for this job
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Save the dataframes
        gstr2b_df.to_pickle(job_dir / "gstr2b_df.pkl")
        pr_df.to_pickle(job_dir / "pr_df.pkl")
        
        # Save the result
        with open(job_dir / "result.json", "w") as f:
            # Convert datetime objects to strings
            result_json = json.dumps(result, default=str)
            f.write(result_json)
        
        logger.info(f"Stored reconciliation data for job {job_id}")
    except Exception as e:
        logger.error(f"Error storing reconciliation data: {e}")

def clean_old_data():
    """Clean up old reconciliation data"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=RETAIN_JOBS_HOURS)
        
        for job_dir in UPLOAD_DIR.iterdir():
            if job_dir.is_dir():
                # Check modification time
                mod_time = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mod_time < cutoff_time:
                    # Remove old directory
                    for file in job_dir.iterdir():
                        file.unlink()
                    job_dir.rmdir()
        
        logger.info("Cleaned up old reconciliation data")
    except Exception as e:
        logger.error(f"Error cleaning old data: {e}")

# ML Feature Engineering Functions
def extract_features(gstr2b_row, pr_row):
    """Extract features for ML model from two invoice records"""
    features = {}
    
    # Text similarity features
    features["invoice_number_similarity"] = fuzz.ratio(
        str(gstr2b_row["invoice_number"]), 
        str(pr_row["invoice_number"])
    ) / 100.0
    
    # Date features
    if isinstance(gstr2b_row["invoice_date"], pd.Timestamp) and isinstance(pr_row["invoice_date"], pd.Timestamp):
        features["date_delta"] = abs((gstr2b_row["invoice_date"] - pr_row["invoice_date"]).days)
    else:
        features["date_delta"] = 30  # Default high value for missing dates
    
    # Amount features
    features["amount_diff_absolute"] = abs(gstr2b_row["total_amount"] - pr_row["total_amount"])
    max_amount = max(gstr2b_row["total_amount"], pr_row["total_amount"], 1)  # Avoid div by zero
    features["amount_diff_percentage"] = features["amount_diff_absolute"] / max_amount
    
    # Tax component features
    features["taxable_value_diff"] = abs(gstr2b_row["taxable_value"] - pr_row["taxable_value"]) / max_amount
    features["igst_diff"] = abs(gstr2b_row["igst"] - pr_row["igst"]) / max_amount if "igst" in gstr2b_row and "igst" in pr_row else 1.0
    features["cgst_diff"] = abs(gstr2b_row["cgst"] - pr_row["cgst"]) / max_amount if "cgst" in gstr2b_row and "cgst" in pr_row else 1.0
    features["sgst_diff"] = abs(gstr2b_row["sgst"] - pr_row["sgst"]) / max_amount if "sgst" in gstr2b_row and "sgst" in pr_row else 1.0
    
    # Same GSTIN feature
    features["same_gstin"] = 1.0 if gstr2b_row["gstin"] == pr_row["gstin"] else 0.0
    
    return features

def create_feature_vector(features_dict):
    """Convert features dictionary to vector"""
    # Ensure consistent feature order
    feature_names = [
        "invoice_number_similarity", "date_delta", "amount_diff_absolute", 
        "amount_diff_percentage", "taxable_value_diff", "igst_diff", 
        "cgst_diff", "sgst_diff", "same_gstin"
    ]
    
    return [features_dict.get(feature, 0.0) for feature in feature_names]

async def load_ml_models():
    """Load ML models if they exist"""
    try:
        # Load matching model
        matching_model_path = MODEL_DIR / "matching_model.pkl"
        if matching_model_path.exists():
            ml_models["matching_model"] = joblib.load(matching_model_path)
            logger.info("Loaded matching model successfully")
        
        # Load anomaly detection model
        anomaly_model_path = MODEL_DIR / "anomaly_model.pkl"
        if anomaly_model_path.exists():
            ml_models["anomaly_model"] = joblib.load(anomaly_model_path)
            logger.info("Loaded anomaly detection model successfully")
        
        # Load text embedding model (for semantic similarity)
        try:
            # Initialize the SBERT model for semantic similarity
            ml_models["embedding_model"] = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logger.info("Loaded text embedding model successfully")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        return False

# Enhanced Invoice Parsing and Standardization
def parse_date(date_str):
    """Parse date string to datetime object with multiple format support."""
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d-%b-%Y", "%d/%b/%Y", "%d %b %Y", "%d %B %Y",
        "%d.%m.%Y", "%Y.%m.%d"  # Added more formats
    ]
    
    # Handle Excel date numbers
    if isinstance(date_str, (int, float)):
        try:
            return pd.to_datetime(date_str, unit='D', origin='1899-12-30').date()
        except:
            pass
    
    # Try parsing with different formats
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except ValueError:
            continue
    
    try:
        # Let pandas try to parse it
        return pd.to_datetime(date_str).date()
    except:
        pass
    
    raise ValueError(f"Could not parse date: {date_str}")

def standardize_invoice_number(invoice_number):
    """Standardize invoice number by removing special characters and spaces."""
    if isinstance(invoice_number, str):
        # Remove common prefixes
        prefixes = ["INV", "INVOICE", "BILL", "BL", "SI", "IN"]
        cleaned = invoice_number.upper()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                if cleaned[len(prefix):len(prefix)+1].isdigit() or cleaned[len(prefix):len(prefix)+1] == "-":
                    cleaned = cleaned[len(prefix):]
                    break
                elif len(cleaned) > len(prefix) and cleaned[len(prefix)] == " ":
                    cleaned = cleaned[len(prefix)+1:]
                    break
        
        # Remove special characters
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in "-/")
        return cleaned
    return str(invoice_number).upper()

def validate_gstin(gstin):
    """Validate GSTIN format."""
    import re
    if not isinstance(gstin, str):
        return False
    
    # Basic GSTIN format: 2 digits, 5 letters, 4 digits, 1 letter, 1 alphanumeric, Z, 1 alphanumeric
    pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$'
    return bool(re.match(pattern, gstin.upper()))

def standardize_gstin(gstin):
    """Standardize GSTIN by removing spaces and converting to uppercase."""
    if isinstance(gstin, str):
        clean_gstin = ''.join(c for c in gstin if c.isalnum()).upper()
        # If not a valid GSTIN format after cleaning, return original standardized
        return clean_gstin
    return str(gstin).upper()

def read_excel_or_csv(file_content, sheet_name=0):
    """Read Excel or CSV file content with enhanced error handling."""
    try:
        # Try reading as Excel
        return pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
    except Exception as e:
        logger.info(f"Not an Excel file or error reading Excel: {e}")
        try:
            # Try reading as CSV with multiple encodings and delimiters
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            delimiters = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        return pd.read_csv(io.BytesIO(file_content), 
                                          encoding=encoding, 
                                          sep=delimiter)
                    except:
                        continue
            
            # Default attempt
            return pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload Excel or CSV file.")

def preprocess_dataframe(df, is_gstr2b=True):
    """Standardize column names and data types with enhanced ML features."""
    # Make column names lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Map expected column names based on file type
    required_columns = {
        'gstin': ['gstin', 'supplier_gstin', 'vendor_gstin', 'counterparty_gstin', 'party_gstin', 'customer_gstin'],
        'invoice_number': ['invoice_number', 'invoice_no', 'bill_number', 'bill_no', 'invoice', 'inv_no'],
        'invoice_date': ['invoice_date', 'bill_date', 'date', 'inv_date'],
        'taxable_value': ['taxable_value', 'taxable_amount', 'taxable', 'tax_base', 'base_amount'],
        'igst': ['igst', 'igst_amount'],
        'cgst': ['cgst', 'cgst_amount'],
        'sgst': ['sgst', 'sgst_amount'],
        'total_tax': ['total_tax', 'tax_amount', 'total_gst', 'gst_amount'],
        'total_amount': ['total_amount', 'invoice_value', 'invoice_amount', 'bill_amount', 'amount', 'total']
    }
    
    # Optional columns that may provide additional ML features
    optional_columns = {
        'description': ['description', 'item_description', 'particulars', 'narration', 'details'],
        'vendor_name': ['vendor_name', 'supplier_name', 'party_name', 'name'],
        'place_of_supply': ['place_of_supply', 'pos', 'supply_place']
    }
    
    # Create a mapping of standard column names to actual column names
    column_mapping = {}
    for std_col, possible_cols in required_columns.items():
        found = False
        for col in possible_cols:
            if col in df.columns:
                column_mapping[std_col] = col
                found = True
                break
        
        if not found:
            if std_col in ['igst', 'cgst', 'sgst', 'total_tax']:
                # For tax columns, default to 0 if not found
                df[std_col] = 0.0
                column_mapping[std_col] = std_col
            elif std_col == 'total_amount':
                # For total amount, try to calculate if not found
                if 'taxable_value' in column_mapping and all(tax in column_mapping for tax in ['igst', 'cgst', 'sgst']):
                    pass  # Will calculate later
                else:
                    raise HTTPException(status_code=400, detail=f"Column {std_col} not found and cannot be calculated")
            else:
                raise HTTPException(status_code=400, detail=f"Required column {std_col} not found in the uploaded file")
    
    # Map optional columns for ML features
    for std_col, possible_cols in optional_columns.items():
        for col in possible_cols:
            if col in df.columns:
                column_mapping[std_col] = col
                break
    
    # Rename columns to standardized names
    df = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Calculate total_amount if not present
    if 'total_amount' not in df.columns:
        if 'total_tax' in df.columns:
            df['total_amount'] = df['taxable_value'] + df['total_tax']
        else:
            df['total_amount'] = df['taxable_value'] + df['igst'] + df['cgst'] + df['sgst']
    
    # Calculate total_tax if not present
    if 'total_tax' not in df.columns:
        df['total_tax'] = df['igst'] + df['cgst'] + df['sgst']
    
    # Ensure numeric columns
    for col in ['taxable_value', 'igst', 'cgst', 'sgst', 'total_tax', 'total_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Standardize date column
    df['invoice_date'] = df['invoice_date'].apply(
        lambda x: parse_date(x) if not pd.isna(x) else None
    )
    
    # Standardize string columns
    df['invoice_number'] = df['invoice_number'].apply(standardize_invoice_number)
    df['gstin'] = df['gstin'].apply(standardize_gstin)
    
    # Add source identifier
    source_type = "GSTR2B" if is_gstr2b else "PR"
    df['source'] = source_type
    
    # Generate unique row ID for tracking
    df['row_id'] = [f"{source_type}_{i}" for i in range(len(df))]
    
    return df

# ML-Enhanced Reconciliation Algorithm
def detect_anomalies(df, sensitivity=0.1):
    """Detect anomalies in invoice data using Isolation Forest"""
    if ml_models["anomaly_model"] is None:
        # Create a new model if none exists
        logger.info("Creating new anomaly detection model")
        model = IsolationForest(contamination=sensitivity, random_state=42)
    else:
        model = ml_models["anomaly_model"]
    
    # Select numeric columns for anomaly detection
    numeric_cols = ['taxable_value', 'igst', 'cgst', 'sgst', 'total_tax', 'total_amount']
    feature_df = df[numeric_cols].copy()
    
    # Handle missing values
    feature_df = feature_df.fillna(0)
    
    # Detect anomalies (-1 for anomalies, 1 for normal)
    if len(feature_df) > 10:  # Only run if we have enough data
        predictions = model.fit_predict(feature_df)
        df['is_anomaly'] = [p == -1 for p in predictions]
    else:
        df['is_anomaly'] = False
        
    return df

def predict_match_probability(features_vector):
    """Predict match probability using the ML model"""
    if ml_models["matching_model"] is None:
        # Fallback to rule-based scoring if no model exists
        features_dict = dict(zip(
            ["invoice_number_similarity", "date_delta", "amount_diff_absolute", 
             "amount_diff_percentage", "taxable_value_diff", "igst_diff", 
             "cgst_diff", "sgst_diff", "same_gstin"],
            features_vector
        ))
        
        # Simple weighted score (similar to the original algorithm)
        score = (
            0.4 * features_dict["invoice_number_similarity"] + 
            0.3 * (1 - min(features_dict["date_delta"] / 30, 1.0)) +
            0.2 * (1 - features_dict["amount_diff_percentage"]) +
            0.1 * features_dict["same_gstin"]
        )
        return float(score)
    else:
        # Use ML model to predict match probability
        try:
            # Get probability of match (class 1)
            proba = ml_models["matching_model"].predict_proba([features_vector])[0][1]
            return float(proba)
        except:
            # If model fails, use a simple rule-based score
            features_dict = dict(zip(
                ["invoice_number_similarity", "date_delta", "amount_diff_absolute", 
                 "amount_diff_percentage", "taxable_value_diff", "igst_diff", 
                 "cgst_diff", "sgst_diff", "same_gstin"],
                features_vector
            ))
            score = (
                0.4 * features_dict["invoice_number_similarity"] + 
                0.3 * (1 - min(features_dict["date_delta"] / 30, 1.0)) +
                0.2 * (1 - features_dict["amount_diff_percentage"]) +
                0.1 * features_dict["same_gstin"]
            )
            return float(score)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two text descriptions"""
    if ml_models["embedding_model"] is None or not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    
    # Clean and normalize texts
    text1 = str(text1).strip().lower()
    text2 = str(text2).strip().lower()
    
    if not text1 or not text2:
        return 0.0
    
    try:
        # Get embeddings
        embedding1 = ml_models["embedding_model"].encode(text1, convert_to_tensor=True)
        embedding2 = ml_models["embedding_model"].encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        from torch import nn
        cos_sim = nn.CosineSimilarity(dim=0)
        similarity = cos_sim(embedding1, embedding2).item()
        
        return float(similarity)
    except Exception as e:
        logger.warning(f"Error calculating semantic similarity: {e}")
        return 0.0

def suggest_correction(gstr2b_row, pr_row):
    """Suggest possible corrections for mismatched invoices"""
    suggestions = []
    
    # Check for date discrepancies
    if "invoice_date" in gstr2b_row and "invoice_date" in pr_row:
        if isinstance(gstr2b_row["invoice_date"], pd.Timestamp) and isinstance(pr_row["invoice_date"], pd.Timestamp):
            date_diff = abs((gstr2b_row["invoice_date"] - pr_row["invoice_date"]).days)
            if date_diff > 0:
                suggestions.append(f"Update invoice date from {pr_row['invoice_date'].strftime('%d-%m-%Y')} to {gstr2b_row['invoice_date'].strftime('%d-%m-%Y')}")
    
    # Check for invoice number formatting issues
    if "invoice_number" in gstr2b_row and "invoice_number" in pr_row:
        if gstr2b_row["invoice_number"] != pr_row["invoice_number"]:
            suggestions.append(f"Update invoice number from {pr_row['invoice_number']} to {gstr2b_row['invoice_number']}")
    
    # Check for amount differences
    if "total_amount" in gstr2b_row and "total_amount" in pr_row:
        amount_diff = gstr2b_row["total_amount"] - pr_row["total_amount"]
        if abs(amount_diff) > 0.1:
            suggestions.append(f"Adjust invoice amount by {amount_diff:.2f} (GSTR-2B: {gstr2b_row['total_amount']:.2f}, PR: {pr_row['total_amount']:.2f})")
    
    if not suggestions:
        return None
    
    return "; ".join(suggestions)

def reconcile_invoices_ml(gstr2b_df, pr_df, config):
    """
    Reconcile GSTR-2B data with Purchase Register data using ML enhancements.
    
    Args:
        gstr2b_df: DataFrame containing GSTR-2B data
        pr_df: DataFrame containing Purchase Register data
        config: Configuration for reconciliation
        
    Returns:
        Dict containing matched, mismatched, and missing invoices
    """
    tolerance_amount = config.tolerance_amount
    tolerance_percentage = config.tolerance_percentage / 100  # Convert to decimal
    fuzzy_match_threshold = config.fuzzy_match_threshold
    date_exact_tolerance = config.date_exact_tolerance
    date_fuzzy_tolerance = config.date_fuzzy_tolerance
    use_ml = config.use_ml and USE_ML
    ml_confidence_threshold = config.ml_confidence_threshold
    detect_anomalies_flag = config.detect_anomalies and USE_ML
    anomaly_sensitivity = config.anomaly_sensitivity
    
    # Create copies to avoid modifying the originals
    gstr2b = gstr2b_df.copy()
    pr = pr_df.copy()
    
    # Detect anomalies if enabled
    if detect_anomalies_flag:
        logger.info("Detecting anomalies")
        gstr2b = detect_anomalies(gstr2b, sensitivity=anomaly_sensitivity)
        pr = detect_anomalies(pr, sensitivity=anomaly_sensitivity)
    else:
        gstr2b['is_anomaly'] = False
        pr['is_anomaly'] = False
    
    # Add flags to track matched invoices
    gstr2b['matched'] = False
    pr['matched'] = False
    
    # Prepare result containers
    matched = []
    mismatched = []
    only_in_gstr2b = []
    only_in_pr = []
    potential_matches = []
    
    # Step 1: Exact Matching (similar to original algorithm)
    for idx, g_row in gstr2b.iterrows():
        # Try to find exact match first
        exact_matches = pr[
            (pr['gstin'] == g_row['gstin']) &
            (pr['invoice_number'] == g_row['invoice_number']) &
            (
                (pr['invoice_date'] == g_row['invoice_date']) |
                (abs((pr['invoice_date'] - g_row['invoice_date']).dt.days) <= date_exact_tolerance)
            )
        ]
        
        if not exact_matches.empty:
            # Found potential exact matches, check amount tolerance
            for p_idx, p_row in exact_matches.iterrows():
                amount_diff = abs(g_row['total_amount'] - p_row['total_amount'])
                within_tolerance = (
                    amount_diff <= tolerance_amount or 
                    amount_diff <= (tolerance_percentage * max(g_row['total_amount'], p_row['total_amount']))
                )
                
                if within_tolerance:
                    # Mark as matched
                    gstr2b.at[idx, 'matched'] = True
                    pr.at[p_idx, 'matched'] = True
                    
                    # Calculate ML confidence for informational purposes
                    ml_confidence = 1.0  # Perfect match gets 100% confidence
                    
                    matched.append({
                        'gstin': g_row['gstin'],
                        'invoice_number': g_row['invoice_number'],
                        'invoice_date': g_row['invoice_date'].strftime('%d-%m-%Y'),
                        'taxable_value': g_row['taxable_value'],
                        'igst': g_row['igst'],
                        'cgst': g_row['cgst'],
                        'sgst': g_row['sgst'],
                        'total_tax': g_row['total_tax'],
                        'total_amount': g_row['total_amount'],
                        'amount_difference': float(amount_diff),
                        'date_delta': 0 if g_row['invoice_date'] == p_row['invoice_date'] else abs((g_row['invoice_date'] - p_row['invoice_date']).days),
                        'ml_confidence': ml_confidence,
                        'is_anomaly': bool(g_row['is_anomaly'] or p_row['is_anomaly'])
                    })
                    break
    
    # Step 2: ML-Enhanced Fuzzy Matching
    for idx, g_row in gstr2b[~gstr2b['matched']].iterrows():
        best_match = None
        best_score = 0
        best_idx = None
        best_features = None
        
        # First filter by GSTIN to reduce comparison space
        potential_matches = pr[
            (pr['gstin'] == g_row['gstin']) & 
            (~pr['matched'])
        ]
        
        # If no matches by GSTIN, try looser matching if ML is enabled
        if potential_matches.empty and use_ml:
            # Try matching without GSTIN restriction
            potential_matches = pr[~pr['matched']]
        
        for p_idx, p_row in potential_matches.iterrows():
            # Extract features for ML model
            features_dict = extract_features(g_row, p_row)
            features_vector = create_feature_vector(features_dict)
            
            # Add semantic similarity if description is available
            desc_similarity = 0.0
            if 'description' in g_row and 'description' in p_row:
                desc_similarity = calculate_semantic_similarity(g_row['description'], p_row['description'])
                features_dict['description_similarity'] = desc_similarity
            
            # Get match probability
            if use_ml:
                match_score = predict_match_probability(features_vector)
            else:
                # Use original scoring method
                inv_score = features_dict["invoice_number_similarity"]
                if inv_score < fuzzy_match_threshold/100:
                    continue
                    
                date_diff = features_dict["date_delta"]
                if date_diff > date_fuzzy_tolerance:
                    continue
                
                # Use weighted score
                match_score = (
                    0.5 * inv_score + 
                    0.3 * (1 - min(date_diff/date_fuzzy_tolerance, 1.0)) + 
                    0.2 * (1 - features_dict["amount_diff_percentage"])
                )
            
            if match_score > best_score:
                best_score = match_score
                best_match = p_row
                best_idx = p_idx
                best_features = features_dict
        
        if best_match is not None and best_score >= ml_confidence_threshold:
            # Determine if this is a match or mismatch
            amount_diff = abs(g_row['total_amount'] - best_match['total_amount'])
            within_tolerance = (
                amount_diff <= tolerance_amount or 
                amount_diff <= (tolerance_percentage * max(g_row['total_amount'], best_match['total_amount']))
            )
            
            date_diff = best_features["date_delta"]
            
            if within_tolerance and date_diff <= date_fuzzy_tolerance:
                # Mark as matched
                gstr2b.at[idx, 'matched'] = True
                pr.at[best_idx, 'matched'] = True
                
                matched.append({
                    'gstin': g_row['gstin'],
                    'invoice_number': g_row['invoice_number'],
                    'invoice_date': g_row['invoice_date'].strftime('%d-%m-%Y'),
                    'taxable_value': g_row['taxable_value'],
                    'igst': g_row['igst'],
                    'cgst': g_row['cgst'],
                    'sgst': g_row['sgst'],
                    'total_tax': g_row['total_tax'],
                    'total_amount': g_row['total_amount'],
                    'match_score': float(best_score),
                    'amount_difference': float(amount_diff),
                    'date_delta': int(date_diff),
                    'ml_confidence': float(best_score),
                    'is_anomaly': bool(g_row['is_anomaly'] or best_match['is_anomaly'])
                })
            else:
                # Mark as mismatched
                gstr2b.at[idx, 'matched'] = True
                pr.at[best_idx, 'matched'] = True
                
                # Generate reason and correction suggestion
                reason = []
                if amount_diff > tolerance_amount and amount_diff > (tolerance_percentage * max(g_row['total_amount'], best_match['total_amount'])):
                    reason.append("Amount mismatch")
                if date_diff > date_exact_tolerance:
                    reason.append("Date mismatch")
                
                # Generate correction suggestion
                correction = suggest_correction(g_row, best_match) if use_ml else None
                
                mismatched.append({
                    'gstin': g_row['gstin'],
                    'invoice_number': g_row['invoice_number'],
                    'invoice_date': g_row['invoice_date'].strftime('%d-%m-%Y'),
                    'taxable_value': g_row['taxable_value'],
                    'igst': g_row['igst'],
                    'cgst': g_row['cgst'],
                    'sgst': g_row['sgst'],
                    'total_tax': g_row['total_tax'],
                    'total_amount': g_row['total_amount'],
                    'match_score': float(best_score),
                    'amount_difference': float(amount_diff),
                    'date_delta': int(date_diff),
                    'reason': ', '.join(reason),
                    'gstr2b_details': g_row.to_dict(),
                    'pr_details': best_match.to_dict(),
                    'ml_confidence': float(best_score),
                    'is_anomaly': bool(g_row['is_anomaly'] or best_match['is_anomaly']),
                    'correction_suggestion': correction
                })
        elif best_match is not None and best_score >= 0.5:  # Lower threshold for potential matches
            # Record as potential match for suggestion
            potential_matches.append({
                'gstr2b_id': g_row['row_id'],
                'pr_id': best_match['row_id'],
                'gstr2b_invoice': g_row['invoice_number'],
                'pr_invoice': best_match['invoice_number'],
                'gstr2b_date': g_row['invoice_date'].strftime('%d-%m-%Y'),
                'pr_date': best_match['invoice_date'].strftime('%d-%m-%Y'),
                'gstr2b_amount': float(g_row['total_amount']),
                'pr_amount': float(best_match['total_amount']),
                'match_score': float(best_score),
                'date_delta': int(best_features["date_delta"]),
                'amount_diff': float(abs(g_row['total_amount'] - best_match['total_amount']))
            })
    
    # Step 3: Collect unmatched invoices
    # Invoices in GSTR-2B but not in Purchase Register
    for idx, g_row in gstr2b[~gstr2b['matched']].iterrows():
        only_in_gstr2b.append({
            'gstin': g_row['gstin'],
            'invoice_number': g_row['invoice_number'],
            'invoice_date': g_row['invoice_date'].strftime('%d-%m-%Y'),
            'taxable_value': g_row['taxable_value'],
            'igst': g_row['igst'],
            'cgst': g_row['cgst'],
            'sgst': g_row['sgst'],
            'total_tax': g_row['total_tax'],
            'total_amount': g_row['total_amount'],
            'is_anomaly': bool(g_row['is_anomaly'])
        })
    
    # Invoices in Purchase Register but not in GSTR-2B
    for idx, p_row in pr[~pr['matched']].iterrows():
        only_in_pr.append({
            'gstin': p_row['gstin'],
            'invoice_number': p_row['invoice_number'],
            'invoice_date': p_row['invoice_date'].strftime('%d-%m-%Y'),
            'taxable_value': p_row['taxable_value'],
            'igst': p_row['igst'],
            'cgst': p_row['cgst'],
            'sgst': p_row['sgst'],
            'total_tax': p_row['total_tax'],
            'total_amount': p_row['total_amount'],
            'is_anomaly': bool(p_row['is_anomaly'])
        })
    
    # Count anomalies
    anomaly_count = (
        sum(1 for item in matched if item.get('is_anomaly', False)) +
        sum(1 for item in mismatched if item.get('is_anomaly', False)) +
        sum(1 for item in only_in_gstr2b if item.get('is_anomaly', False)) +
        sum(1 for item in only_in_pr if item.get('is_anomaly', False))
    )
    
    # Generate summary
    summary = {
        'total_gstr2b_invoices': len(gstr2b),
        'total_purchase_invoices': len(pr),
        'matched_count': len(matched),
        'mismatched_count': len(mismatched),
        'missing_in_gstr2b': len(only_in_pr),
        'missing_in_purchase_register': len(only_in_gstr2b),
        'anomaly_count': anomaly_count
    }
    
    result = {
        'matched': matched,
        'mismatched': mismatched,
        'only_in_gstr2b': only_in_gstr2b,
        'only_in_purchase_register': only_in_pr,
        'potential_matches': potential_matches if use_ml else [],
        'summary': summary
    }
    
    return result

# ML Model Training Functions
async def collect_training_data():
    """Collect historical reconciliation data for ML training"""
    all_gstr2b = []
    all_pr = []
    match_pairs = []
    
    try:
        # Iterate through job directories
        for job_dir in UPLOAD_DIR.iterdir():
            if not job_dir.is_dir():
                continue
                
            gstr2b_path = job_dir / "gstr2b_df.pkl"
            pr_path = job_dir / "pr_df.pkl"
            result_path = job_dir / "result.json"
            
            if not all(p.exists() for p in [gstr2b_path, pr_path, result_path]):
                continue
                
            # Load the data
            try:
                gstr2b_df = pd.read_pickle(gstr2b_path)
                pr_df = pd.read_pickle(pr_path)
                
                with open(result_path, 'r') as f:
                    result = json.load(f)
                
                # Add to collection
                all_gstr2b.append(gstr2b_df)
                all_pr.append(pr_df)
                
                # Extract match pairs
                for match in result.get('matched', []):
                    g_row = gstr2b_df[gstr2b_df['invoice_number'] == match['invoice_number']].iloc[0]
                    p_candidates = pr_df[pr_df['invoice_number'] == match['invoice_number']]
                    
                    if not p_candidates.empty:
                        p_row = p_candidates.iloc[0]
                        features = extract_features(g_row, p_row)
                        match_pairs.append((features, 1))  # 1 = match
                
                # Extract non-match pairs (sample a few)
                for _ in range(min(len(result.get('matched', [])), 10)):
                    g_idx = np.random.randint(0, len(gstr2b_df))
                    p_idx = np.random.randint(0, len(pr_df))
                    
                    g_row = gstr2b_df.iloc[g_idx]
                    p_row = pr_df.iloc[p_idx]
                    
                    # Check if this is not a known match
                    inv_match = g_row['invoice_number'] == p_row['invoice_number']
                    
                    if not inv_match:
                        features = extract_features(g_row, p_row)
                        match_pairs.append((features, 0))  # 0 = non-match
            except Exception as e:
                logger.warning(f"Error processing job data: {e}")
                continue
        
        return all_gstr2b, all_pr, match_pairs
    except Exception as e:
        logger.error(f"Error collecting training data: {e}")
        return [], [], []

async def train_matching_model(match_pairs):
    """Train the invoice matching model"""
    try:
        if not match_pairs:
            logger.warning("No training data available for matching model")
            return None, {}
            
        # Prepare data
        X = np.array([create_feature_vector(pair[0]) for pair in match_pairs])
        y = np.array([pair[1] for pair in match_pairs])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        
        return model, metrics
    except Exception as e:
        logger.error(f"Error training matching model: {e}")
        return None, {}

async def train_anomaly_model(all_gstr2b, all_pr):
    """Train the anomaly detection model"""
    try:
        if not all_gstr2b or not all_pr:
            logger.warning("No training data available for anomaly model")
            return None, {}
            
        # Combine all dataframes
        gstr2b_combined = pd.concat(all_gstr2b, ignore_index=True)
        pr_combined = pd.concat(all_pr, ignore_index=True)
        all_data = pd.concat([gstr2b_combined, pr_combined], ignore_index=True)
        
        # Select numeric columns for anomaly detection
        numeric_cols = ['taxable_value', 'igst', 'cgst', 'sgst', 'total_tax', 'total_amount']
        feature_df = all_data[numeric_cols].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Train model
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(feature_df)
        
        # Simple evaluation (percentage of anomalies)
        predictions = model.predict(feature_df)
        anomaly_ratio = (predictions == -1).mean()
        
        metrics = {
            'anomaly_ratio': float(anomaly_ratio),
            'feature_importance': {col: 1.0 for col in numeric_cols}  # Not applicable for IsolationForest
        }
        
        return model, metrics
    except Exception as e:
        logger.error(f"Error training anomaly model: {e}")
        return None, {}

async def process_ml_training(job_id, config):
    """Process ML model training in background"""
    try:
        # Update job status
        set_job_status(job_id, {
            "status": "processing", 
            "message": "Collecting training data"
        })
        
        # Collect training data
        all_gstr2b, all_pr, match_pairs = await collect_training_data()
        
        metrics = {}
        
        # Train matching model if requested
        if config.retrain_matching_model:
            set_job_status(job_id, {
                "status": "processing", 
                "message": "Training matching model"
            })
            
            model, model_metrics = await train_matching_model(match_pairs)
            if model:
                # Save model
                joblib.dump(model, MODEL_DIR / "matching_model.pkl")
                ml_models["matching_model"] = model
                metrics["matching_model"] = model_metrics
                logger.info("Matching model trained and saved successfully")
        
        # Train anomaly model if requested
        if config.retrain_anomaly_model:
            set_job_status(job_id, {
                "status": "processing", 
                "message": "Training anomaly detection model"
            })
            
            model, model_metrics = await train_anomaly_model(all_gstr2b, all_pr)
            if model:
                # Save model
                joblib.dump(model, MODEL_DIR / "anomaly_model.pkl")
                ml_models["anomaly_model"] = model
                metrics["anomaly_model"] = model_metrics
                logger.info("Anomaly detection model trained and saved successfully")
        
        # Update job status to completed
        set_job_status(job_id, {
            "status": "completed", 
            "message": "ML training completed successfully",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error during ML training: {str(e)}")
        set_job_status(job_id, {
            "status": "failed", 
            "message": f"ML training failed: {str(e)}"
        })

def process_reconciliation_ml(gstr2b_content, pr_content, config: ReconciliationRequest):
    """Process reconciliation with ML enhancements."""
    try:
        # Read and preprocess data
        gstr2b_df = preprocess_dataframe(read_excel_or_csv(gstr2b_content), is_gstr2b=True)
        pr_df = preprocess_dataframe(read_excel_or_csv(pr_content), is_gstr2b=False)
        
        # Perform reconciliation with ML
        result = reconcile_invoices_ml(gstr2b_df, pr_df, config)
        
        return result, gstr2b_df, pr_df
    except Exception as e:
        logger.error(f"Error during reconciliation: {str(e)}")
        raise e

def process_job_ml(job_id: str, gstr2b_content: bytes, pr_content: bytes, config: ReconciliationRequest):
    """Process reconciliation job with ML and update status."""
    try:
        # Update job status to processing
        set_job_status(job_id, {"status": "processing", "message": "Reconciliation in progress"})
        
        # Process reconciliation
        result, gstr2b_df, pr_df = process_reconciliation_ml(gstr2b_content, pr_content, config)
        
        # Store data for future training if ML is enabled
        if USE_ML and config.use_ml:
            store_reconciliation_data(job_id, gstr2b_df, pr_df, result)
        
        # Update job status to completed
        set_job_status(job_id, {
            "status": "completed", 
            "message": "Reconciliation completed successfully",
            "result": result
        })
    except Exception as e:
        # Update job status to failed
        logger.error(f"Job {job_id} failed: {str(e)}")
        set_job_status(job_id, {"status": "failed", "message": f"Error: {str(e)}"})

# FastAPI routes
@app.post("/api/reconcile", response_model=JobStatus)
async def reconcile_files(
    background_tasks: BackgroundTasks,
    gstr2b_file: UploadFile = File(...),
    purchase_register_file: UploadFile = File(...),
    config: ReconciliationRequest = Depends()
):
    """
    Reconcile GSTR-2B data with Purchase Register data with ML enhancements.
    
    - **gstr2b_file**: GSTR-2B Excel or CSV file
    - **purchase_register_file**: Purchase Register Excel or CSV file
    - **config**: Reconciliation configuration with ML options
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Read file contents
        gstr2b_content = await gstr2b_file.read()
        purchase_register_content = await purchase_register_file.read()
        
        # Initialize job status
        set_job_status(job_id, {"status": "queued", "message": "Job queued for processing"})
        
        # Process reconciliation in background
        background_tasks.add_task(
            process_job_ml, 
            job_id, 
            gstr2b_content, 
            purchase_register_content, 
            config
        )
        
        return {"job_id": job_id, "status": "queued", "message": "Reconciliation job queued for processing"}
    
    except Exception as e:
        logger.error(f"Error submitting reconciliation job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/job-status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a reconciliation job."""
    job_status = get_job_status(job_id)
    
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job_status.get("status", "unknown"),
        "message": job_status.get("message", ""),
        "result": job_status.get("result")
    }

@app.post("/api/train-ml", response_model=MLTrainingStatus)
async def train_ml_models(
    background_tasks: BackgroundTasks,
    config: MLTrainingRequest
):
    """
    Train ML models for invoice matching and anomaly detection.
    
    - **config**: ML training configuration
    """
    try:
        # Check if ML is enabled
        if not USE_ML:
            raise HTTPException(status_code=400, detail="ML features are disabled")
            
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        set_job_status(job_id, {"status": "queued", "message": "ML training job queued"})
        
        # Process training in background
        background_tasks.add_task(process_ml_training, job_id, config)
        
        return {
            "job_id": job_id, 
            "status": "queued", 
            "message": "ML training job queued for processing"
        }
    
    except Exception as e:
        logger.error(f"Error submitting ML training job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/feedback", response_model=dict)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for ML model improvement.
    
    - **feedback**: Feedback data for model improvement
    """
    try:
        # Get the job result
        job_status = get_job_status(feedback.job_id)
        
        if job_status is None or "result" not in job_status:
            raise HTTPException(status_code=404, detail="Job result not found")
            
        # Store feedback for future training
        feedback_dir = MODEL_DIR / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / f"{feedback.job_id}_{uuid.uuid4()}.json"
        with open(feedback_file, "w") as f:
            json.dump(feedback.dict(), f)
        
        return {"status": "success", "message": "Feedback recorded successfully"}
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/ml-status")
async def get_ml_status():
    """Get status of ML features and models."""
    return {
        "ml_enabled": USE_ML,
        "models": {
            "matching_model": "loaded" if ml_models["matching_model"] is not None else "not_loaded",
            "anomaly_model": "loaded" if ml_models["anomaly_model"] is not None else "not_loaded",
            "embedding_model": "loaded" if ml_models["embedding_model"] is not None else "not_loaded"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "GST Reconciliation ML API is running"}

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Load ML models
    if USE_ML:
        await load_ml_models()
    
    # Start clean up task
    async def cleanup_task():
        while True:
            clean_old_data()
            await asyncio.sleep(3600)  # Run every hour
    
    asyncio.create_task(cleanup_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)