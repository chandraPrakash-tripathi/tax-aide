from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import os
import logging
import uuid

from app.utils.file_handlers import FileHandler
from app.services.data_processor import DataProcessor
from app.services.cache_service import CacheService
from app.api.dependencies import get_current_user_id
from app.models.schemas import UploadResponse, FileValidationResponse
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
data_processor = DataProcessor()
cache_service = CacheService()

@router.post("/upload/gst2b", response_model=UploadResponse)
async def upload_gst2b_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """
    Upload and validate GSTR-2B JSON file
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension != '.json':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a JSON file."
        )
    
    try:
        # Save the uploaded file
        file_path = await FileHandler.save_upload_file(file, user_id, "gst2b")
        
        # Validate the file
        is_valid = FileHandler.validate_json_file(file_path)
        if not is_valid:
            # Remove invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid GSTR-2B JSON format"
            )
        
        # Cache the file path
        file_info = cache_service.get(f"files:{user_id}") or {}
        file_info["gst2b"] = file_path
        cache_service.set(f"files:{user_id}", file_info)
        
        return UploadResponse(
            filename=file.filename,
            file_path=file_path,
            file_type="gst2b",
            is_valid=True,
            message="GSTR-2B file uploaded successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing GSTR-2B upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}"
        )

@router.post("/upload/purchase", response_model=UploadResponse)
async def upload_purchase_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """
    Upload and validate purchase register CSV file
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension != '.csv':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
    
    try:
        # Save the uploaded file
        file_path = await FileHandler.save_upload_file(file, user_id, "purchase")
        
        # Validate the file
        is_valid = FileHandler.validate_csv_file(file_path)
        if not is_valid:
            # Remove invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid purchase register CSV format"
            )
        
        # Cache the file path
        file_info = cache_service.get(f"files:{user_id}") or {}
        file_info["purchase"] = file_path
        cache_service.set(f"files:{user_id}", file_info)
        
        return UploadResponse(
            filename=file.filename,
            file_path=file_path,
            file_type="purchase",
            is_valid=True,
            message="Purchase register file uploaded successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing purchase register upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}"
        )

@router.post("/validate-files", response_model=FileValidationResponse)
async def validate_uploaded_files(
    user_id: str = Depends(get_current_user_id)
):
    """
    Validate both uploaded files for reconciliation
    """
    try:
        # Get cached file paths
        file_info = cache_service.get(f"files:{user_id}") or {}
        
        gst2b_path = file_info.get("gst2b")
        purchase_path = file_info.get("purchase")
        
        # Check if both files are uploaded
        if not gst2b_path or not purchase_path:
            return FileValidationResponse(
                is_valid=False,
                gst2b_file_present=bool(gst2b_path),
                purchase_file_present=bool(purchase_path),
                message="Please upload both GSTR-2B and purchase register files"
            )
        
        # Validate if files exist
        gst2b_exists = os.path.exists(gst2b_path)
        purchase_exists = os.path.exists(purchase_path)
        
        if not gst2b_exists or not purchase_exists:
            # Update cache if files don't exist
            if not gst2b_exists:
                file_info.pop("gst2b", None)
            if not purchase_exists:
                file_info.pop("purchase", None)
            
            cache_service.set(f"files:{user_id}", file_info)
            
            return FileValidationResponse(
                is_valid=False,
                gst2b_file_present=gst2b_exists,
                purchase_file_present=purchase_exists,
                message="One or more files are missing. Please upload again."
            )
        
        # Additional validation can be done here if needed
        
        return FileValidationResponse(
            is_valid=True,
            gst2b_file_present=True,
            purchase_file_present=True,
            message="Files are ready for reconciliation"
        )
    
    except Exception as e:
        logger.error(f"Error validating files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate files: {str(e)}"
        )