import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.config import settings
from app.models.schemas import GstrFile, MatchStatus, ReconciliationResult
from app.core.matching import (
    exact_match_invoices, 
    fuzzy_match_invoices, 
    semantic_match_invoices
)
from app.utils.helpers import (
    normalize_date, 
    normalize_invoice_number, 
    normalize_decimal
)

logger = logging.getLogger(__name__)

class ReconciliationEngine:
    """
    Core engine for reconciling GSTR-2B data with purchase register
    """
    
    def __init__(
        self, 
        gstr2b_data: Dict[str, Any], 
        purchase_data: pd.DataFrame,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
        use_ml: bool = False
    ):
        self.gstr2b_data = GstrFile(**gstr2b_data)
        self.purchase_df = self._preprocess_purchase_data(purchase_data)
        self.similarity_threshold = similarity_threshold
        self.use_ml = use_ml
        self.gstr2b_df = self._convert_gstr2b_to_dataframe()
        
    def _preprocess_purchase_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess purchase register data"""
        # Rename columns to standardized format
        column_mapping = {
            'GSTIN of Supplier': 'gstin',
            'Invoice Number': 'invoice_number',
            'Invoice date': 'invoice_date',
            'Invoice Value': 'invoice_value',
            'Place Of Supply': 'place_of_supply',
            'Reverse Charge': 'reverse_charge',
            'Invoice Type': 'invoice_type',
            'Rate': 'tax_rate',
            'Taxable Value': 'taxable_value',
            'Integrated Tax Paid': 'igst',
            'Central Tax Paid': 'cgst',
            'State/UT Tax Paid': 'sgst',
            'Status': 'status'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Normalize data types
        df['gstin'] = df['gstin'].str.strip().str.upper()
        df['invoice_number'] = df['invoice_number'].apply(normalize_invoice_number)
        
        # Handle date conversion
        df['invoice_date'] = df['invoice_date'].apply(
            lambda x: normalize_date(x, input_format=settings.CSV_DATE_FORMAT)
        )
        
        # Convert numeric fields
        numeric_cols = ['invoice_value', 'taxable_value', 'igst', 'cgst', 'sgst', 'tax_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalize_decimal)
        
        # Add match status column if not present
        if 'status' not in df.columns:
            df['status'] = None
            
        # Add hash column for detecting duplicates
        df['invoice_hash'] = df.apply(
            lambda row: f"{row['gstin']}_{row['invoice_number']}_{row['invoice_date']}", 
            axis=1
        )
        
        # Flag duplicates
        df['is_duplicate'] = df.duplicated(subset=['invoice_hash'], keep='first')
        
        return df
    
    def _convert_gstr2b_to_dataframe(self) -> pd.DataFrame:
        """Convert GSTR-2B JSON data to DataFrame for easier processing"""
        records = []
        
        for supplier in self.gstr2b_data.data.docdata.b2b:
            for invoice in supplier.inv:
                # Calculate total taxable value and tax amounts
                total_taxable_value = sum(item.txval for item in invoice.items)
                total_sgst = sum(item.sgst for item in invoice.items)
                total_cgst = sum(item.cgst for item in invoice.items)
                total_cess = sum(item.cess for item in invoice.items)
                total_igst = 0  # IGST not directly available in items
                
                # Extract tax rates
                tax_rates = set(item.rt for item in invoice.items)
                tax_rate = list(tax_rates)[0] if len(tax_rates) == 1 else 0
                
                record = {
                    'gstin': supplier.ctin,
                    'supplier_name': supplier.trdnm,
                    'invoice_number': normalize_invoice_number(invoice.inum),
                    'invoice_date': invoice.dt,
                    'invoice_value': invoice.val,
                    'place_of_supply': invoice.pos,
                    'reverse_charge': invoice.rev,
                    'invoice_type': invoice.typ,
                    'tax_rate': tax_rate,
                    'taxable_value': total_taxable_value,
                    'igst': total_igst,
                    'cgst': total_cgst,
                    'sgst': total_sgst,
                    'cess': total_cess,
                    'itc_available': invoice.itcavl,
                    'supply_period': supplier.supprd,
                    'status': invoice.status.value if invoice.status else None,
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add hash column for lookup
        if not df.empty:
            df['invoice_hash'] = df.apply(
                lambda row: f"{row['gstin']}_{row['invoice_number']}_{row['invoice_date']}", 
                axis=1
            )
        
        return df
    
    def reconcile(self) -> ReconciliationResult:
        """
        Perform reconciliation between GSTR-2B and purchase register
        """
        logger.info("Starting reconciliation process")
        
        # 1. Perform exact matching first
        self._perform_exact_matching()
        
        # 2. Perform fuzzy matching on unmatched entries
        self._perform_fuzzy_matching()
        
        # 3. If ML is enabled, perform semantic matching on remaining unmatched entries
        if self.use_ml:
            self._perform_semantic_matching()
        
        # 4. Identify missing invoices in each data source
        self._identify_missing_invoices()
        
        # 5. Compile results
        result = self._compile_results()
        
        logger.info(f"Reconciliation completed with {len(result.matched)} matches, "
                   f"{len(result.unmatched)} unmatched, "
                   f"{len(result.missing_in_csv)} missing in CSV, "
                   f"{len(result.missing_in_json)} missing in JSON")
        
        return result
        
    def _perform_exact_matching(self):
        """Match invoices based on exact matches of key fields"""
        exact_matches = exact_match_invoices(self.gstr2b_df, self.purchase_df)
        
        # Update status for matches
        for match_idx in exact_matches:
            gstr_idx, purchase_idx = match_idx
            
            # Update GSTR2B status
            self.gstr2b_df.loc[gstr_idx, 'status'] = MatchStatus.MATCH.value
            self.gstr2b_df.loc[gstr_idx, 'matched_purchase_idx'] = purchase_idx
            
            # Update Purchase status
            self.purchase_df.loc[purchase_idx, 'status'] = MatchStatus.MATCH.value
            self.purchase_df.loc[purchase_idx, 'matched_gstr_idx'] = gstr_idx
    
    def _perform_fuzzy_matching(self):
        """Match invoices using fuzzy matching for unmatched entries"""
        # Filter unmatched entries
        gstr_unmatched = self.gstr2b_df[
            (self.gstr2b_df['status'].isna()) | 
            (self.gstr2b_df['status'] != MatchStatus.MATCH.value)
        ]
        
        purchase_unmatched = self.purchase_df[
            (self.purchase_df['status'].isna()) | 
            (self.purchase_df['status'] != MatchStatus.MATCH.value)
        ]
        
        # Skip if either dataset is empty
        if gstr_unmatched.empty or purchase_unmatched.empty:
            return
        
        # Perform fuzzy matching
        fuzzy_matches = fuzzy_match_invoices(
            gstr_unmatched, 
            purchase_unmatched,
            threshold=settings.FUZZY_THRESHOLD
        )
        
        # Update status for fuzzy matches
        for match_idx in fuzzy_matches:
            gstr_idx, purchase_idx = match_idx
            
            # Update GSTR2B status
            self.gstr2b_df.loc[gstr_idx, 'status'] = MatchStatus.MATCH.value
            self.gstr2b_df.loc[gstr_idx, 'matched_purchase_idx'] = purchase_idx
            self.gstr2b_df.loc[gstr_idx, 'match_type'] = 'fuzzy'
            
            # Update Purchase status
            self.purchase_df.loc[purchase_idx, 'status'] = MatchStatus.MATCH.value
            self.purchase_df.loc[purchase_idx, 'matched_gstr_idx'] = gstr_idx
            self.purchase_df.loc[purchase_idx, 'match_type'] = 'fuzzy'
    
    def _perform_semantic_matching(self):
        """Match invoices using semantic similarity for remaining unmatched entries"""
        # Filter unmatched entries
        gstr_unmatched = self.gstr2b_df[
            (self.gstr2b_df['status'].isna()) | 
            (self.gstr2b_df['status'] != MatchStatus.MATCH.value)
        ]
        
        purchase_unmatched = self.purchase_df[
            (self.purchase_df['status'].isna()) | 
            (self.purchase_df['status'] != MatchStatus.MATCH.value)
        ]
        
        # Skip if either dataset is empty
        if gstr_unmatched.empty or purchase_unmatched.empty:
            return
        
        # Perform semantic matching
        semantic_matches = semantic_match_invoices(
            gstr_unmatched,
            purchase_unmatched,
            threshold=self.similarity_threshold
        )
        
        # Update status for semantic matches
        for match_idx in semantic_matches:
            gstr_idx, purchase_idx = match_idx
            
            # Update GSTR2B status
            self.gstr2b_df.loc[gstr_idx, 'status'] = MatchStatus.MATCH.value
            self.gstr2b_df.loc[gstr_idx, 'matched_purchase_idx'] = purchase_idx
            self.gstr2b_df.loc[gstr_idx, 'match_type'] = 'semantic'
            
            # Update Purchase status
            self.purchase_df.loc[purchase_idx, 'status'] = MatchStatus.MATCH.value
            self.purchase_df.loc[purchase_idx, 'matched_gstr_idx'] = gstr_idx
            self.purchase_df.loc[purchase_idx, 'match_type'] = 'semantic'
    
    def _identify_missing_invoices(self):
        """Identify invoices missing in either GSTR-2B or purchase register"""
        # Mark invoices missing in purchase register
        gstr_unmatched = self.gstr2b_df[
            (self.gstr2b_df['status'].isna()) | 
            (self.gstr2b_df['status'] != MatchStatus.MATCH.value)
        ]
        
        for idx in gstr_unmatched.index:
            self.gstr2b_df.loc[idx, 'status'] = MatchStatus.NOT_IN_CSV.value
        
        # Mark invoices missing in GSTR-2B
        purchase_unmatched = self.purchase_df[
            (self.purchase_df['status'].isna()) | 
            (self.purchase_df['status'] != MatchStatus.MATCH.value)
        ]
        
        for idx in purchase_unmatched.index:
            if self.purchase_df.loc[idx, 'is_duplicate']:
                self.purchase_df.loc[idx, 'status'] = MatchStatus.DUPLICATE.value
            else:
                self.purchase_df.loc[idx, 'status'] = MatchStatus.NOT_IN_JSON.value
    
    def _compile_results(self) -> ReconciliationResult:
        """Compile reconciliation results"""
        # Matched invoices
        matched = []
        
        for idx, row in self.purchase_df[self.purchase_df['status'] == MatchStatus.MATCH.value].iterrows():
            gstr_idx = row.get('matched_gstr_idx')
            if pd.notna(gstr_idx):
                gstr_row = self.gstr2b_df.loc[gstr_idx].to_dict()
                purchase_row = row.to_dict()
                
                match_data = {
                    'purchase': {k: v for k, v in purchase_row.items() if k != 'matched_gstr_idx'},
                    'gstr2b': {k: v for k, v in gstr_row.items() if k != 'matched_purchase_idx'},
                    'match_type': purchase_row.get('match_type', 'exact'),
                    'differences': self._get_differences(purchase_row, gstr_row)
                }
                
                matched.append(match_data)
        
        # Unmatched invoices (status is UNMATCH)
        unmatched = []
        for idx, row in self.purchase_df[self.purchase_df['status'] == MatchStatus.UNMATCH.value].iterrows():
            unmatched.append(row.to_dict())
        
        # Missing in CSV (present in GSTR-2B but not in purchase register)
        missing_in_csv = []
        for idx, row in self.gstr2b_df[self.gstr2b_df['status'] == MatchStatus.NOT_IN_CSV.value].iterrows():
            missing_in_csv.append(row.to_dict())
        
        # Missing in JSON (present in purchase register but not in GSTR-2B)
        missing_in_json = []
        for idx, row in self.purchase_df[self.purchase_df['status'] == MatchStatus.NOT_IN_JSON.value].iterrows():
            missing_in_json.append(row.to_dict())
        
        # Duplicates
        duplicates = []
        for idx, row in self.purchase_df[self.purchase_df['status'] == MatchStatus.DUPLICATE.value].iterrows():
            duplicates.append(row.to_dict())
        
        # Summary statistics
        summary = self._generate_summary()
        
        return ReconciliationResult(
            matched=matched,
            unmatched=unmatched,
            missing_in_csv=missing_in_csv,
            missing_in_json=missing_in_json,
            duplicates=duplicates,
            summary=summary
        )
    
    def _get_differences(self, purchase_row: Dict[str, Any], gstr_row: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between matched purchase and GSTR-2B entries"""
        differences = {}
        
        # Fields to compare
        fields_to_compare = {
            'taxable_value': 'Taxable Value',
            'invoice_value': 'Invoice Value',
            'cgst': 'CGST',
            'sgst': 'SGST',
            'igst': 'IGST'
        }
        
        for field, label in fields_to_compare.items():
            purchase_value = purchase_row.get(field, 0) or 0
            gstr_value = gstr_row.get(field, 0) or 0
            
            if abs(float(purchase_value) - float(gstr_value)) > 0.01:  # Allow small difference due to floating point precision
                differences[field] = {
                    'label': label,
                    'purchase_value': float(purchase_value),
                    'gstr_value': float(gstr_value),
                    'difference': float(purchase_value) - float(gstr_value)
                }
        
        return differences
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for reconciliation"""
        # Count statistics
        total_purchase_invoices = len(self.purchase_df)
        total_gstr_invoices = len(self.gstr2b_df)
        
        matched_count = len(self.purchase_df[self.purchase_df['status'] == MatchStatus.MATCH.value])
        unmatched_count = len(self.purchase_df[self.purchase_df['status'] == MatchStatus.UNMATCH.value])
        missing_in_csv_count = len(self.gstr2b_df[self.gstr2b_df['status'] == MatchStatus.NOT_IN_CSV.value])
        missing_in_json_count = len(self.purchase_df[self.purchase_df['status'] == MatchStatus.NOT_IN_JSON.value])
        duplicate_count = len(self.purchase_df[self.purchase_df['status'] == MatchStatus.DUPLICATE.value])
        
        # Calculate value statistics
        matched_purchase_value = self.purchase_df[self.purchase_df['status'] == MatchStatus.MATCH.value]['taxable_value'].sum()
        matched_gstr_value = sum(
            self.gstr2b_df.loc[idx]['taxable_value'] 
            for idx in self.purchase_df[self.purchase_df['status'] == MatchStatus.MATCH.value]['matched_gstr_idx']
            if pd.notna(idx) and idx in self.gstr2b_df.index
        )
        
        unmatched_value = self.purchase_df[self.purchase_df['status'] == MatchStatus.UNMATCH.value]['taxable_value'].sum()
        missing_in_csv_value = self.gstr2b_df[self.gstr2b_df['status'] == MatchStatus.NOT_IN_CSV.value]['taxable_value'].sum()
        missing_in_json_value = self.purchase_df[self.purchase_df['status'] == MatchStatus.NOT_IN_JSON.value]['taxable_value'].sum()
        
        # Tax implications
        potential_itc_loss = self.purchase_df[self.purchase_df['status'] == MatchStatus.NOT_IN_JSON.value].apply(
            lambda row: row['cgst'] + row['sgst'] + row['igst'], axis=1
        ).sum()
        
        return {
            "total_purchase_invoices": total_purchase_invoices,
            "total_gstr_invoices": total_gstr_invoices,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "missing_in_csv_count": missing_in_csv_count,
            "missing_in_json_count": missing_in_json_count,
            "duplicate_count": duplicate_count,
            "matched_purchase_value": float(matched_purchase_value),
            "matched_gstr_value": float(matched_gstr_value),
            "matched_value_difference": float(matched_purchase_value - matched_gstr_value),
            "unmatched_value": float(unmatched_value),
            "missing_in_csv_value": float(missing_in_csv_value),
            "missing_in_json_value": float(missing_in_json_value),
            "potential_itc_loss": float(potential_itc_loss),
            "match_percentage": round((matched_count / total_purchase_invoices * 100) if total_purchase_invoices > 0 else 0, 2)
        }