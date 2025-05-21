import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from app.models.schemas import ReconciliationResult
from app.utils.file_handlers import FileHandler

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class to generate reconciliation reports"""
    
    @staticmethod
    def prepare_reconciliation_dataframe(gst2b_df: pd.DataFrame, purchase_df: pd.DataFrame, 
                                         match_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare a final dataframe for reconciliation reporting
        
        Args:
            gst2b_df: DataFrame with GSTR-2B data
            purchase_df: DataFrame with purchase register data
            match_results: Results from the matching process
            
        Returns:
            pd.DataFrame: Combined DataFrame with all reconciliation data
        """
        # Initialize result dataframe
        result_df = pd.DataFrame()
        
        # Process matched invoices
        matched_rows = []
        for match in match_results.get('matches', []):
            if 'gst2b_idx' in match and 'purchase_idx' in match:
                gst_row = gst2b_df.iloc[match['gst2b_idx']].copy()
                purchase_row = purchase_df.iloc[match['purchase_idx']].copy()
                
                # Add match status
                gst_row['match_status'] = 'match'
                purchase_row['match_status'] = 'match'
                
                # Calculate total tax
                gst_row['total_tax'] = float(gst_row.get('cgst', 0)) + float(gst_row.get('sgst', 0)) + float(gst_row.get('igst', 0))
                purchase_row['total_tax'] = float(purchase_row.get('cgst', 0)) + float(purchase_row.get('sgst', 0)) + float(purchase_row.get('igst', 0))
                
                matched_rows.append(gst_row)
                matched_rows.append(purchase_row)
        
        # Process unmatched invoices from GSTR-2B
        for idx in match_results.get('unmatched_gst2b', []):
            row = gst2b_df.iloc[idx].copy()
            row['match_status'] = 'not in csv'
            row['total_tax'] = float(row.get('cgst', 0)) + float(row.get('sgst', 0)) + float(row.get('igst', 0))
            matched_rows.append(row)
        
        # Process unmatched invoices from purchase register
        for idx in match_results.get('unmatched_purchase', []):
            row = purchase_df.iloc[idx].copy()
            row['match_status'] = 'not in json'
            row['total_tax'] = float(row.get('cgst', 0)) + float(row.get('sgst', 0)) + float(row.get('igst', 0))
            matched_rows.append(row)
        
        # Process partial matches
        for match in match_results.get('partial_matches', []):
            if 'gst2b_idx' in match and 'purchase_idx' in match:
                gst_row = gst2b_df.iloc[match['gst2b_idx']].copy()
                purchase_row = purchase_df.iloc[match['purchase_idx']].copy()
                
                # Add match status
                gst_row['match_status'] = 'unmatch'
                purchase_row['match_status'] = 'unmatch'
                
                # Calculate total tax
                gst_row['total_tax'] = float(gst_row.get('cgst', 0)) + float(gst_row.get('sgst', 0)) + float(gst_row.get('igst', 0))
                purchase_row['total_tax'] = float(purchase_row.get('cgst', 0)) + float(purchase_row.get('sgst', 0)) + float(purchase_row.get('igst', 0))
                
                matched_rows.append(gst_row)
                matched_rows.append(purchase_row)
        
        # Combine all rows
        if matched_rows:
            result_df = pd.DataFrame(matched_rows)
        
        return result_df
    
    @staticmethod
    def calculate_summary_statistics(result_df: pd.DataFrame) -> ReconciliationResult:
        """
        Calculate summary statistics from reconciliation results
        
        Args:
            result_df: DataFrame with reconciliation results
            
        Returns:
            ReconciliationResult: Summary of reconciliation results
        """
        # Initialize counters
        total_invoices = len(result_df)
        matched_invoices = len(result_df[result_df['match_status'] == 'match'])
        unmatched_invoices = len(result_df[result_df['match_status'] == 'unmatch'])
        missing_in_gst2b = len(result_df[result_df['match_status'] == 'not in json'])
        missing_in_purchase = len(result_df[result_df['match_status'] == 'not in csv'])
        
        # Filter for GST2B and purchase records
        gst2b_records = result_df[result_df['source'] == 'gst2b']
        purchase_records = result_df[result_df['source'] == 'purchase']
        
        # Calculate tax totals
        total_tax_in_gst2b = gst2b_records['total_tax'].sum() if 'total_tax' in gst2b_records.columns else 0
        total_tax_in_purchase = purchase_records['total_tax'].sum() if 'total_tax' in purchase_records.columns else 0
        
        # Calculate eligible ITC
        eligible_itc = result_df[
            (result_df['match_status'] == 'match') & 
            (result_df['source'] == 'gst2b')
        ]['total_tax'].sum() if 'total_tax' in result_df.columns else 0
        
        # Calculate excess tax claimed and unclaimed tax
        excess_tax_claimed = result_df[
            (result_df['match_status'] == 'not in json') & 
            (result_df['source'] == 'purchase')
        ]['total_tax'].sum() if 'total_tax' in result_df.columns else 0
        
        unclaimed_tax = result_df[
            (result_df['match_status'] == 'not in csv') & 
            (result_df['source'] == 'gst2b')
        ]['total_tax'].sum() if 'total_tax' in result_df.columns else 0
        
        # Return summary object
        return ReconciliationResult(
            total_invoices=total_invoices,
            matched_invoices=matched_invoices,
            unmatched_invoices=unmatched_invoices,
            missing_in_gst2b=missing_in_gst2b,
            missing_in_purchase=missing_in_purchase,
            total_tax_in_gst2b=total_tax_in_gst2b,
            total_tax_in_purchase=total_tax_in_purchase,
            eligible_itc=eligible_itc,
            excess_tax_claimed=excess_tax_claimed,
            unclaimed_tax=unclaimed_tax
        )
    
    @staticmethod
    def generate_excel_report(result_df: pd.DataFrame, summary: ReconciliationResult, 
                              output_path: str) -> str:
        """
        Generate an Excel report from reconciliation results
        
        Args:
            result_df: DataFrame with reconciliation results
            summary: Summary of reconciliation results
            output_path: Path to save the report
            
        Returns:
            str: Path to the generated Excel file
        """
        # Convert summary to dictionary for the file handler
        summary_dict = summary.dict() if hasattr(summary, 'dict') else vars(summary)
        
        # Use the file handler to generate Excel report
        return FileHandler.generate_excel_report(result_df, summary_dict, output_path)
    
    @staticmethod
    def generate_csv_report(result_df: pd.DataFrame, output_path: str) -> str:
        """
        Generate a CSV report from reconciliation results
        
        Args:
            result_df: DataFrame with reconciliation results
            output_path: Path to save the report
            
        Returns:
            str: Path to the generated CSV file
        """
        return FileHandler.export_to_csv(result_df, output_path)
    
    @staticmethod
    def generate_json_report(result_df: pd.DataFrame, summary: ReconciliationResult, 
                            output_path: str) -> str:
        """
        Generate a JSON report from reconciliation results
        
        Args:
            result_df: DataFrame with reconciliation results
            summary: Summary of reconciliation results
            output_path: Path to save the report
            
        Returns:
            str: Path to the generated JSON file
        """
        import json
        import os
        
        os.makedirs(output_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reconciliation_report_{timestamp}.json"
        file_path = os.path.join(output_path, filename)
        
        # Convert summary to dictionary
        summary_dict = summary.dict() if hasattr(summary, 'dict') else vars(summary)
        
        # Convert DataFrame to records
        records = result_df.to_dict(orient='records')
        
        # Create report JSON
        report_data = {
            "summary": summary_dict,
            "records": records,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report generated: {file_path}")
        return file_path