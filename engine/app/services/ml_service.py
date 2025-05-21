from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import logging
from fuzzywuzzy import fuzz

from app.config import settings

logger = logging.getLogger(__name__)

class MLService:
    """Service for machine learning functionality in GST reconciliation"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.text_encoder = None
        self._load_models()
    
    def _load_models(self):
        """Load the ML models if they exist, otherwise initialize new ones"""
        try:
            model_dir = settings.ML_MODEL_DIR
            
            # Try to load XGBoost model if it exists
            model_path = os.path.join(model_dir, "invoice_matcher.model")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Loaded XGBoost model from disk")
            
            # Try to load scaler if it exists
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded StandardScaler from disk")
            else:
                self.scaler = StandardScaler()
            
            # Initialize sentence transformer for text embeddings
            try:
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded Sentence Transformer model")
            except Exception as e:
                logger.warning(f"Failed to load Sentence Transformer: {str(e)}")
                self.text_encoder = None
                
        except Exception as e:
            logger.warning(f"Failed to load ML models: {str(e)}")
            self.model = None
            self.scaler = None
    
    def save_models(self):
        """Save the trained models to disk"""
        try:
            model_dir = settings.ML_MODEL_DIR
            os.makedirs(model_dir, exist_ok=True)
            
            # Save XGBoost model
            if self.model:
                model_path = os.path.join(model_dir, "invoice_matcher.model")
                joblib.dump(self.model, model_path)
                
            # Save scaler
            if self.scaler:
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                
            logger.info("Successfully saved ML models to disk")
        except Exception as e:
            logger.error(f"Failed to save ML models: {str(e)}")
    
    def generate_features(self, row_gst: pd.Series, row_purchase: pd.Series) -> np.ndarray:
        """Generate features for invoice matching based on two rows of data"""
        features = []
        
        # GSTIN similarity
        gstin_similarity = 1.0 if row_gst['gstin'] == row_purchase['gstin'] else 0.0
        features.append(gstin_similarity)
        
        # Invoice number similarity
        inv_similarity = fuzz.ratio(str(row_gst['invoice_number']), str(row_purchase['invoice_number'])) / 100.0
        features.append(inv_similarity)
        
        # Invoice date similarity (0 if not the same, 1 if the same)
        date_similarity = 1.0 if row_gst['invoice_date'] == row_purchase['invoice_date'] else 0.0
        features.append(date_similarity)
        
        # Taxable value difference ratio
        gst_taxable = float(row_gst['taxable_value'])
        purchase_taxable = float(row_purchase['taxable_value'])
        
        if gst_taxable > 0 and purchase_taxable > 0:
            value_diff_ratio = min(gst_taxable, purchase_taxable) / max(gst_taxable, purchase_taxable)
        else:
            value_diff_ratio = 0.0
        features.append(value_diff_ratio)
        
        # Tax amount difference ratio
        gst_tax = float(row_gst.get('cgst', 0)) + float(row_gst.get('sgst', 0)) + float(row_gst.get('igst', 0))
        purchase_tax = float(row_purchase.get('cgst', 0)) + float(row_purchase.get('sgst', 0)) + float(row_purchase.get('igst', 0))
        
        if gst_tax > 0 and purchase_tax > 0:
            tax_diff_ratio = min(gst_tax, purchase_tax) / max(gst_tax, purchase_tax)
        else:
            tax_diff_ratio = 0.0
        features.append(tax_diff_ratio)
        
        # Use text embeddings if available
        if self.text_encoder:
            try:
                # Use invoice number as text feature
                gst_inv_embedding = self.text_encoder.encode(str(row_gst['invoice_number']))
                purchase_inv_embedding = self.text_encoder.encode(str(row_purchase['invoice_number']))
                
                # Calculate cosine similarity between embeddings
                cosine_sim = np.dot(gst_inv_embedding, purchase_inv_embedding) / (
                    np.linalg.norm(gst_inv_embedding) * np.linalg.norm(purchase_inv_embedding)
                )
                features.append(cosine_sim)
            except Exception as e:
                # Fall back to zero if encoding fails
                logger.warning(f"Text encoding failed: {str(e)}")
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_match(self, row_gst: pd.Series, row_purchase: pd.Series) -> float:
        """Predict whether two invoice entries match"""
        features = self.generate_features(row_gst, row_purchase)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.model:
            # Use the ML model if available
            prediction = self.model.predict_proba(features)[0, 1]  # Probability of matching
        else:
            # Fall back to rule-based approach if no model is available
            gstin_match = row_gst['gstin'] == row_purchase['gstin']
            invoice_match = fuzz.ratio(str(row_gst['invoice_number']), str(row_purchase['invoice_number'])) > 90
            date_match = row_gst['invoice_date'] == row_purchase['invoice_date']
            
            gst_taxable = float(row_gst['taxable_value'])
            purchase_taxable = float(row_purchase['taxable_value'])
            value_match = abs(gst_taxable - purchase_taxable) / max(gst_taxable, purchase_taxable) < 0.05 if max(gst_taxable, purchase_taxable) > 0 else False
            
            # Simple rule-based scoring
            score = 0.0
            if gstin_match: score += 0.4
            if invoice_match: score += 0.3
            if date_match: score += 0.2
            if value_match: score += 0.1
            
            prediction = score
        
        return prediction
    
    def train_model(self, training_data: pd.DataFrame):
        """Train the XGBoost model using labeled data"""
        if len(training_data) < 10:
            logger.warning("Not enough training data to train model")
            return False
        
        try:
            # Extract features and labels
            X = training_data.drop(['match_label'], axis=1).values
            y = training_data['match_label'].values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100
            )
            self.model.fit(X_scaled, y)
            
            # Save the trained model
            self.save_models()
            
            logger.info("Successfully trained invoice matching model")
            return True
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            return False
    
    def generate_training_data(self, gst2b_df: pd.DataFrame, purchase_df: pd.DataFrame, 
                               matches: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate training data from matched and unmatched invoices"""
        training_rows = []
        
        # Process known matches
        for match in matches:
            if 'gst2b_idx' in match and 'purchase_idx' in match:
                gst_row = gst2b_df.iloc[match['gst2b_idx']]
                purchase_row = purchase_df.iloc[match['purchase_idx']]
                
                features = self.generate_features(gst_row, purchase_row)[0]
                
                # Positive example (match)
                training_rows.append(np.append(features, 1))
                
                # Find a negative example (non-match) for this row
                # Get a random purchase row that is not the matching one
                random_idx = np.random.randint(0, len(purchase_df))
                while random_idx == match['purchase_idx']:
                    random_idx = np.random.randint(0, len(purchase_df))
                
                random_purchase = purchase_df.iloc[random_idx]
                neg_features = self.generate_features(gst_row, random_purchase)[0]
                
                # Negative example (non-match)
                training_rows.append(np.append(neg_features, 0))
        
        # Create DataFrame from training rows
        if training_rows:
            # Create feature names based on the number of features
            feature_names = [f'feature_{i}' for i in range(len(training_rows[0])-1)]
            feature_names.append('match_label')
            
            training_df = pd.DataFrame(training_rows, columns=feature_names)
            return training_df
        else:
            # Return empty DataFrame with proper columns if no training data
            columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'match_label']
            return pd.DataFrame(columns=columns)