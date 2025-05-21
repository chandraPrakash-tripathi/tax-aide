import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

def random_gstin():
    """Generate a random GSTIN."""
    state_code = random.randint(1, 36)
    pan = ''.join(random.choices(string.ascii_uppercase, k=5)) + ''.join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase)
    entity_code = random.randint(1, 9)
    check_digit = random.choice(string.ascii_uppercase + string.digits)
    
    return f"{state_code:02d}{pan}{entity_code}{check_digit}"

def random_invoice_number(length=6):
    """Generate a random invoice number."""
    prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
    number = ''.join(random.choices(string.digits, k=length-2))
    
    return f"{prefix}{number}"

def generate_gstr2b_data(num_records=100):
    """Generate sample GSTR-2B data."""
    gstins = [random_gstin() for _ in range(num_records // 5)]  # Use fewer GSTINs for realistic data
    
    data = []
    for _ in range(num_records):
        gstin = random.choice(gstins)
        invoice_number = random_invoice_number()
        invoice_date = datetime.now().date() - timedelta(days=random.randint(1, 60))
        taxable_value = round(random.uniform(1000, 100000), 2)
        
        # Determine tax rates (0%, 5%, 12%, 18%, 28%)
        tax_rate = random.choice([0, 0.05, 0.12, 0.18, 0.28])
        
        # Calculate tax amounts
        igst = 0
        cgst = 0
        sgst = 0
        
        # Randomly choose between IGST and CGST+SGST
        if random.choice([True, False]):
            igst = round(taxable_value * tax_rate, 2)
        else:
            cgst = round(taxable_value * tax_rate / 2, 2)
            sgst = round(taxable_value * tax_rate / 2, 2)
        
        total_tax = igst + cgst + sgst
        total_amount = taxable_value + total_tax
        
        data.append({
            'GSTIN': gstin,
            'Invoice Number': invoice_number,
            'Invoice Date': invoice_date,
            'Taxable Value': taxable_value,
            'IGST': igst,
            'CGST': cgst,
            'SGST': sgst,
            'Total Tax': total_tax,
            'Total Amount': total_amount
        })
    
    return pd.DataFrame(data)

def generate_purchase_register_data(gstr2b_df, mismatch_rate=0.1, missing_rate=0.05, extra_rate=0.1):
    """
    Generate Purchase Register data based on GSTR-2B data with deliberate mismatches.
    
    Args:
        gstr2b_df: DataFrame containing GSTR-2B data
        mismatch_rate: Percentage of invoices to have mismatches
        missing_rate: Percentage of GSTR-2B invoices to be missing from PR
        extra_rate: Percentage of extra invoices in PR not in GSTR-2B
        
    Returns:
        DataFrame containing Purchase Register data
    """
    # Make a copy of GSTR-2B data
    pr_df = gstr2b_df.copy()
    
    # Number of records in GSTR-2B
    num_records = len(gstr2b_df)
    
    # Calculate number of invoices for each category
    num_mismatches = int(num_records * mismatch_rate)
    num_missing = int(num_records * missing_rate)
    num_extra = int(num_records * extra_rate)
    
    # Randomly select invoices for mismatches
    mismatch_indices = random.sample(range(num_records), num_mismatches)
    
    # Randomly select invoices to be missing from PR
    missing_indices = random.sample([i for i in range(num_records) if i not in mismatch_indices], num_missing)
    
    # Apply mismatches
    for idx in mismatch_indices:
        # Randomly choose type of mismatch
        mismatch_type = random.choice(['amount', 'date', 'invoice_number'])
        
        if mismatch_type == 'amount':
            # Modify amount by a small percentage
            factor = random.uniform(0.95, 1.05)
            pr_df.at[idx, 'Taxable Value'] = round(pr_df.at[idx, 'Taxable Value'] * factor, 2)
            
            # Recalculate taxes and total
            if pr_df.at[idx, 'IGST'] > 0:
                tax_rate = pr_df.at[idx, 'IGST'] / gstr2b_df.at[idx, 'Taxable Value']
                pr_df.at[idx, 'IGST'] = round(pr_df.at[idx, 'Taxable Value'] * tax_rate, 2)
            else:
                tax_rate = (pr_df.at[idx, 'CGST'] + pr_df.at[idx, 'SGST']) / gstr2b_df.at[idx, 'Taxable Value']
                pr_df.at[idx, 'CGST'] = round(pr_df.at[idx, 'Taxable Value'] * tax_rate / 2, 2)
                pr_df.at[idx, 'SGST'] = round(pr_df.at[idx, 'Taxable Value'] * tax_rate / 2, 2)
            
            pr_df.at[idx, 'Total Tax'] = pr_df.at[idx, 'IGST'] + pr_df.at[idx, 'CGST'] + pr_df.at[idx, 'SGST']
            pr_df.at[idx, 'Total Amount'] = pr_df.at[idx, 'Taxable Value'] + pr_df.at[idx, 'Total Tax']
        
        elif mismatch_type == 'date':
            # Modify date by a few days
            days_shift = random.choice([-7, -3, 3, 7])
            pr_df.at[idx, 'Invoice Date'] = pr_df.at[idx, 'Invoice Date']