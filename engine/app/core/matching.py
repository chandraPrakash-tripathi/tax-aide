import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict, Any
import logging
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import joblib
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.utils.helpers import normalize_invoice_number

logger = logging.getLogger(__name__)

def exact_match_invoices(
    gstr_df: pd.DataFrame, 
    purchase_df: pd.DataFrame
) -> List[Tuple[int, int]]:
    """
    Match invoices using exact matching on key fields
    
    Args:
        gstr_df: DataFrame containing GSTR-2B data
        purchase_df: DataFrame containing purchase register data
        
    Returns:
        List of tuples (gstr_index, purchase_index) for matched records
    """
    logger.info("Starting exact matching process")
    
    if gstr_df.empty or purchase_df.empty:
        logger.warning("One of the dataframes is empty, cannot perform matching")
        return []
    
    # Create dictionaries for faster lookup
    purchase_dict = {}
    for idx, row in purchase_df.iterrows():
        key = (
            row['gstin'], 
            row['invoice_number'], 
            row['invoice_date']
        )
        purchase_dict[key] = idx
    
    matches = []
    
    # Find exact matches
    for idx, row in gstr_df.iterrows():
        key = (
            row['gstin'], 
            row['invoice_number'], 
            row['invoice_date']
        )
        
        if key in purchase_dict:
            purchase_idx = purchase_dict[key]
            
            # Further verify that taxable values and tax amounts match closely
            purchase_row = purchase_df.loc[purchase_idx]
            
            # Check if taxable values are within 1% of each other
            gstr_taxable = row['taxable_value'] if pd.notna(row['taxable_value']) else 0
            purchase_taxable = purchase_row['taxable_value'] if pd.notna(purchase_row['taxable_value']) else 0
            
            # Avoid division by zero
            if gstr_taxable > 0 and purchase_taxable > 0:
                taxable_diff_pct = abs(gstr_taxable - purchase_taxable) / max(gstr_taxable, purchase_taxable)
                
                if taxable_diff_pct <= 0.01:  # Within 1% tolerance
                    matches.append((idx, purchase_idx))
                else:
                    # Match but with differences
                    logger.debug(f"Found matching invoice with taxable value difference: {row['invoice_number']}")
                    matches.append((idx, purchase_idx))
            else:
                # If either value is zero, check if both are zero
                if gstr_taxable == 0 and purchase_taxable == 0:
                    matches.append((idx, purchase_idx))
                else:
                    logger.debug(f"One taxable value is zero: {row['invoice_number']}")
    
    logger.info(f"Exact matching completed: found {len(matches)} matches")
    return matches

def fuzzy_match_invoices(
    gstr_df: pd.DataFrame, 
    purchase_df: pd.DataFrame,
    threshold: int = 85
) -> List[Tuple[int, int]]:
    """
    Match invoices using fuzzy matching on invoice numbers and other fields
    
    Args:
        gstr_df: DataFrame containing GSTR-2B data
        purchase_df: DataFrame containing purchase register data
        threshold: Minimum fuzzy match score (0-100)
        
    Returns:
        List of tuples (gstr_index, purchase_index) for matched records
    """
    logger.info("Starting fuzzy matching process")
    
    if gstr_df.empty or purchase_df.empty:
        logger.warning("One of the dataframes is empty, cannot perform fuzzy matching")
        return []
    
    matches = []
    
    # Group by supplier GSTIN to reduce search space
    for gstin, gstr_supplier_df in gstr_df.groupby('gstin'):
        # Find corresponding purchase records for this supplier
        purchase_supplier_df = purchase_df[purchase_df['gstin'] == gstin]
        
        if purchase_supplier_df.empty:
            continue
        
        # For each GSTR invoice, try to find a fuzzy match in purchase invoices
        for gstr_idx, gstr_row in gstr_supplier_df.iterrows():
            best_match_idx = None
            best_match_score = 0
            
            for purchase_idx, purchase_row in purchase_supplier_df.iterrows():
                # Only compare invoice numbers if dates are within a reasonable range
                gstr_date = datetime.strptime(gstr_row['invoice_date'], '%d-%m-%Y')
                purchase_date = datetime.strptime(purchase_row['invoice_date'], '%d-%m-%Y')
                
                # Allow for dates within 3 days difference - vendors might record different dates
                if abs((gstr_date - purchase_date).days) <= 3:
                    # Compute fuzzy match score for invoice number
                    score = fuzz.token_sort_ratio(
                        str(gstr_row['invoice_number']).lower(), 
                        str(purchase_row['invoice_number']).lower()
                    )
                    
                    # Check if taxable values are within 10% of each other
                    gstr_taxable = gstr_row['taxable_value'] if pd.notna(gstr_row['taxable_value']) else 0
                    purchase_taxable = purchase_row['taxable_value'] if pd.notna(purchase_row['taxable_value']) else 0
                    
                    # Check reasonable value range - ignore matches with huge value differences
                    if max(gstr_taxable, purchase_taxable) > 0:
                        value_ratio = min(gstr_taxable, purchase_taxable) / max(gstr_taxable, purchase_taxable)
                        
                        # If values and invoice numbers are close, consider it a match
                        if score > best_match_score and score >= threshold and value_ratio >= 0.8:
                            best_match_score = score
                            best_match_idx = purchase_idx
            
            if best_match_idx is not None:
                matches.append((gstr_idx, best_match_idx))
    
    logger.info(f"Fuzzy matching completed: found {len(matches)} matches")
    return matches

# Cache the model loading to avoid reloading for each call
@lru_cache(maxsize=1)
def _get_sentence_transformer():
    """Load the sentence transformer model with caching"""
    try:
        return SentenceTransformer(settings.EMBEDDINGS_MODEL)
    except Exception as e:
        logger.error(f"Error loading sentence transformer model: {e}")
        return None

def semantic_match_invoices(
    gstr_df: pd.DataFrame, 
    purchase_df: pd.DataFrame,
    threshold: float = 0.85
) -> List[Tuple[int, int]]:
    """
    Match invoices using semantic similarity on invoice features
    
    Args:
        gstr_df: DataFrame containing GSTR-2B data
        purchase_df: DataFrame containing purchase register data
        threshold: Minimum similarity score (0-1)
        
    Returns:
        List of tuples (gstr_index, purchase_index) for matched records
    """
    logger.info("Starting semantic matching process")
    
    if gstr_df.empty or purchase_df.empty:
        logger.warning("One of the dataframes is empty, cannot perform semantic matching")
        return []
    
    # Try to load the model, if it fails, return empty matches
    model = _get_sentence_transformer()
    if model is None:
        logger.error("Could not load semantic model, skipping semantic matching")
        return []
    
    matches = []
    
    # Group by supplier GSTIN to reduce search space
    for gstin, gstr_supplier_df in gstr_df.groupby('gstin'):
        # Find corresponding purchase records for this supplier
        purchase_supplier_df = purchase_df[purchase_df['gstin'] == gstin]
        
        if purchase_supplier_df.empty:
            continue
        
        # Create feature vectors for each invoice
        gstr_features = []
        purchase_features = []
        
        for idx, row in gstr_supplier_df.iterrows():
            # Create a feature string representing this invoice
            feature_str = f"Invoice {row['invoice_number']} on {row['invoice_date']} for {row['taxable_value']:.2f} with tax {row.get('cgst', 0) + row.get('sgst', 0) + row.get('igst', 0):.2f}"
            gstr_features.append((idx, feature_str))
        
        for idx, row in purchase_supplier_df.iterrows():
            feature_str = f"Invoice {row['invoice_number']} on {row['invoice_date']} for {row['taxable_value']:.2f} with tax {row.get('cgst', 0) + row.get('sgst', 0) + row.get('igst', 0):.2f}"
            purchase_features.append((idx, feature_str))
        
        # Extract the feature strings
        gstr_texts = [f[1] for f in gstr_features]
        purchase_texts = [f[1] for f in purchase_features]
        
        if not gstr_texts or not purchase_texts:
            continue
        
        # Get embeddings
        try:
            gstr_embeddings = model.encode(gstr_texts)
            purchase_embeddings = model.encode(purchase_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(gstr_embeddings, purchase_embeddings)
            
            # Find best matches above threshold
            for i, gstr_idx in enumerate(gstr_supplier_df.index):
                best_match_j = np.argmax(similarity_matrix[i])
                best_match_score = similarity_matrix[i][best_match_j]
                
                if best_match_score >= threshold:
                    purchase_idx = purchase_supplier_df.index[best_match_j]
                    matches.append((gstr_idx, purchase_idx))
        
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
    
    logger.info(f"Semantic matching completed: found {len(matches)} matches")
    return matches

def ml_enhanced_matching(
    gstr_df: pd.DataFrame, 
    purchase_df: pd.DataFrame,
    model_path: str = settings.ML_MODEL_PATH
) -> List[Tuple[int, int]]:
    """
    Use trained ML model to predict matches between GSTR-2B and purchase data
    
    Args:
        gstr_df: DataFrame containing GSTR-2B data
        purchase_df: DataFrame containing purchase register data
        model_path: Path to the saved ML model
        
    Returns:
        List of tuples (gstr_index, purchase_index) for matched records
    """
    logger.info("Starting ML-enhanced matching process")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Prepare candidate pairs
        candidate_pairs = []
        candidate_indices = []
        
        # Group by supplier GSTIN to reduce search space
        for gstin, gstr_supplier_df in gstr_df.groupby('gstin'):
            purchase_supplier_df = purchase_df[purchase_df['gstin'] == gstin]
            
            if purchase_supplier_df.empty:
                continue
            
            for gstr_idx, gstr_row in gstr_supplier_df.iterrows():
                for purchase_idx, purchase_row in purchase_supplier_df.iterrows():
                    # Create feature vector for this pair
                    features = create_matching_features(gstr_row, purchase_row)
                    candidate_pairs.append(features)
                    candidate_indices.append((gstr_idx, purchase_idx))
        
        if not candidate_pairs:
            return []
        
        # Convert to numpy array
        X = np.array(candidate_pairs)
        
        # Predict probabilities
        y_proba = model.predict_proba(X)
        
        # Get matches above threshold
        matches = []
        for i, (gstr_idx, purchase_idx) in enumerate(candidate_indices):
            if y_proba[i, 1] >= 0.8:  # Probability of being a match >= 80%
                matches.append((gstr_idx, purchase_idx))
        
        logger.info(f"ML-enhanced matching completed: found {len(matches)} matches")
        return matches
        
    except Exception as e:
        logger.error(f"Error in ML-enhanced matching: {e}")
        return []

def create_matching_features(gstr_row: pd.Series, purchase_row: pd.Series) -> List[float]:
    """Create feature vector for ML model from two invoice rows"""
    features = []
    
    # String similarity features
    invoice_sim = fuzz.token_sort_ratio(
        str(gstr_row.get('invoice_number', '')), 
        str(purchase_row.get('invoice_number', ''))
    ) / 100.0
    
    # Date difference in days
    try:
        gstr_date = datetime.strptime(gstr_row['invoice_date'], '%d-%m-%Y')
        purchase_date = datetime.strptime(purchase_row['invoice_date'], '%d-%m-%Y')
        date_diff = abs((gstr_date - purchase_date).days)
    except:
        date_diff = 30  # Maximum difference if parsing fails
    
    # Normalize date diff
    date_diff_norm = min(date_diff / 30.0, 1.0)
    
    # Value differences
    gstr_taxable = gstr_row.get('taxable_value', 0) or 0
    purchase_taxable = purchase_row.get('taxable_value', 0) or 0
    
    if max(gstr_taxable, purchase_taxable) > 0:
        taxable_ratio = min(gstr_taxable, purchase_taxable) / max(gstr_taxable, purchase_taxable)
    else:
        taxable_ratio = 1.0 if gstr_taxable == purchase_taxable else 0.0
    
    # Tax amounts
    gstr_tax = (gstr_row.get('cgst', 0) or 0) + (gstr_row.get('sgst', 0) or 0) + (gstr_row.get('igst', 0) or 0)
    purchase_tax = (purchase_row.get('cgst', 0) or 0) + (purchase_row.get('sgst', 0) or 0) + (purchase_row.get('igst', 0) or 0)
    
    if max(gstr_tax, purchase_tax) > 0:
        tax_ratio = min(gstr_tax, purchase_tax) / max(gstr_tax, purchase_tax)
    else:
        tax_ratio = 1.0 if gstr_tax == purchase_tax else 0.0
    
    # Tax rate match
    tax_rate_match = 1.0 if gstr_row.get('tax_rate') == purchase_row.get('tax_rate') else 0.0
    
    # Combine features
    features = [
        invoice_sim,
        1.0 - date_diff_norm,  # Higher is better
        taxable_ratio,
        tax_ratio,
        tax_rate_match
    ]
    
    return features