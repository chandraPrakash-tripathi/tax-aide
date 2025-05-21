from typing import List, Tuple
import json
import pandas as pd
import logging
from datetime import datetime

from app.utils.helpers import normalize_date, normalize_gstin, normalize_invoice_number

from app.models.schemas import (
    GstrFile, GstrInvoice, GstrSupplier, GstrData,
    PurchaseInvoice, ReconciliationSummary, MatchStatus
)

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.gst2b_invoices: List[GstrInvoice] = []
        self.purchase_invoices: List[PurchaseInvoice] = []

    def load_gst2b_data(self, file_path: str) -> List[GstrInvoice]:
        """Load and parse GSTR-2B JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            gstr_file = GstrFile(**data)
            suppliers = gstr_file.data.docdata.b2b

            all_invoices = []
            for supplier in suppliers:
                for inv in supplier.inv:
                    inv.status = None  # Can be assigned later during reconciliation
                    all_invoices.append(inv)

            self.gst2b_invoices = all_invoices
            return all_invoices

        except Exception as e:
            logger.error(f"Failed to load GSTR-2B data: {e}")
            raise ValueError(f"Failed to process GST 2B data: {str(e)}")

    def load_purchase_data(self, file_path: str) -> List[PurchaseInvoice]:
        """Load and parse purchase register from CSV file"""
        try:
            df = pd.read_csv(file_path)

            df['GSTIN of Supplier'] = df['GSTIN of Supplier'].apply(normalize_gstin)
            df['Invoice Number'] = df['Invoice Number'].apply(normalize_invoice_number)
            df['Invoice date'] = df['Invoice date'].apply(normalize_date)

            numeric_columns = ['Invoice Value', 'Taxable Value', 'Integrated Tax Paid',
                               'Central Tax Paid', 'State/UT Tax Paid']

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            invoice_objs = []
            for _, row in df.iterrows():
                invoice = PurchaseInvoice(
                    gstin=row['GSTIN of Supplier'],
                    invoice_number=row['Invoice Number'],
                    invoice_date=row['Invoice date'],
                    invoice_value=row.get('Invoice Value', 0.0),
                    place_of_supply=row.get('Place Of Supply', ''),
                    reverse_charge=row.get('Reverse Charge', 'N'),
                    invoice_type=row.get('Invoice Type', ''),
                    tax_rate=row.get('Rate', 0.0),
                    taxable_value=row.get('Taxable Value', 0.0),
                    igst=row.get('Integrated Tax Paid', 0.0),
                    cgst=row.get('Central Tax Paid', 0.0),
                    sgst=row.get('State/UT Tax Paid', 0.0),
                )
                invoice_objs.append(invoice)

            self.purchase_invoices = invoice_objs
            return invoice_objs

        except Exception as e:
            logger.error(f"Failed to load purchase register: {e}")
            raise ValueError(f"Failed to process purchase data: {str(e)}")

    def prepare_data_for_reconciliation(self) -> Tuple[List[GstrInvoice], List[PurchaseInvoice]]:
        """Return the loaded data in a standard format for reconciliation"""
        if not self.gst2b_invoices or not self.purchase_invoices:
            raise ValueError("Both GST 2B and purchase data must be loaded before reconciliation")
        return self.gst2b_invoices, self.purchase_invoices

    def generate_reconciliation_summary(self, result_df: pd.DataFrame) -> ReconciliationSummary:
        """Summarize the reconciliation result"""
        total_invoices = len(result_df)
        matched = len(result_df[result_df['match_status'] == MatchStatus.MATCH])
        unmatched = len(result_df[result_df['match_status'] == MatchStatus.UNMATCH])
        not_in_json = len(result_df[result_df['match_status'] == MatchStatus.NOT_IN_JSON])
        not_in_csv = len(result_df[result_df['match_status'] == MatchStatus.NOT_IN_CSV])
        duplicates = len(result_df[result_df['match_status'] == MatchStatus.DUPLICATE])

        total_tax_json = result_df[result_df['source'] == 'gst2b']['total_tax'].sum()
        total_tax_csv = result_df[result_df['source'] == 'purchase']['total_tax'].sum()

        eligible_itc = result_df[
            (result_df['match_status'] == MatchStatus.MATCH) &
            (result_df['source'] == 'gst2b')
        ]['total_tax'].sum()

        excess_claim = result_df[
            (result_df['match_status'] == MatchStatus.NOT_IN_JSON) &
            (result_df['source'] == 'purchase')
        ]['total_tax'].sum()

        unclaimed = result_df[
            (result_df['match_status'] == MatchStatus.NOT_IN_CSV) &
            (result_df['source'] == 'gst2b')
        ]['total_tax'].sum()

        return ReconciliationSummary(
            total_invoices=total_invoices,
            matched_invoices=matched,
            unmatched_invoices=unmatched,
            missing_in_gst2b=not_in_json,
            missing_in_purchase=not_in_csv,
            total_tax_in_gst2b=total_tax_json,
            total_tax_in_purchase=total_tax_csv,
            eligible_itc=eligible_itc,
            excess_tax_claimed=excess_claim,
            unclaimed_tax=unclaimed
        )
