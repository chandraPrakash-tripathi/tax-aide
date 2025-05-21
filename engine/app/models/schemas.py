from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal

from pydantic import BaseModel, Field, validator, root_validator
from decimal import Decimal


class MatchStatus(str, Enum):
    MATCH = "match"
    UNMATCH = "unmatch"
    NOT_IN_JSON = "not in json"
    NOT_IN_CSV = "not in csv"
    DUPLICATE = "duplicate"


class InvoiceItem(BaseModel):
    sgst: float
    rt: float  # tax rate
    num: int  # HSN/SAC code
    txval: float  # taxable value
    cgst: float
    cess: float = 0


class GstrInvoice(BaseModel):
    dt: str  # Invoice date
    val: float  # Invoice value
    rev: str  # Reverse charge
    itcavl: str  # ITC available
    diffprcnt: float
    pos: str  # Place of supply
    typ: str  # Invoice type
    inum: str  # Invoice number
    rsn: str = ""  # Reason
    status: Optional[MatchStatus] = None
    items: List[InvoiceItem]
    
    @property
    def invoice_date(self) -> datetime:
        """Convert string date to datetime object"""
        return datetime.strptime(self.dt, "%d-%m-%Y")
    
    @property
    def taxable_value(self) -> float:
        """Calculate total taxable value from all items"""
        return sum(item.txval for item in self.items)
    
    @property
    def total_tax(self) -> float:
        """Calculate total tax from all items"""
        return sum(item.sgst + item.cgst + item.cess for item in self.items)


class GstrSupplier(BaseModel):
    inv: List[GstrInvoice]
    trdnm: str  # Trade name
    supfildt: str  # Supply file date
    supprd: str  # Supply period
    ctin: str  # GSTIN
    
    @property
    def gstin(self) -> str:
        return self.ctin


class Gstr2b(BaseModel):
    b2b: List[GstrSupplier] = []


class DocumentData(BaseModel):
    b2b: List[GstrSupplier] = []


class ItcSummary(BaseModel):
    sgst: float
    cgst: float
    cess: float
    igst: float


class B2bSummary(BaseModel):
    sgst: float
    cgst: float
    cess: float
    igst: float


class NonRevSup(BaseModel):
    sgst: float
    b2b: B2bSummary
    cgst: float
    cess: float
    igst: float


class ItcAvl(BaseModel):
    nonrevsup: NonRevSup


class ItcSumm(BaseModel):
    itcavl: ItcAvl


class GstrData(BaseModel):
    itcsumm: ItcSumm
    rtnprd: str  # Return period
    docdata: DocumentData
    gendt: str  # Generation date
    gstin: str  # GSTIN of the business
    version: str


class GstrFile(BaseModel):
    data: GstrData
    chksum: str  # Checksum


class PurchaseInvoice(BaseModel):
    gstin: str  # GSTIN of supplier
    invoice_number: str
    invoice_date: str
    invoice_value: float
    place_of_supply: str
    reverse_charge: str
    invoice_type: str
    tax_rate: float
    taxable_value: float
    igst: float
    cgst: float
    sgst: float
    status: Optional[MatchStatus] = None
    
    @validator('invoice_date', pre=True)
    def parse_date(cls, v):
        """Ensure consistent date format"""
        if isinstance(v, datetime):
            return v.strftime("%d-%m-%Y")
        return v


class ReconciliationResult(BaseModel):
    matched: List[Dict[str, Any]] = Field(default_factory=list)
    unmatched: List[Dict[str, Any]] = Field(default_factory=list)
    missing_in_csv: List[Dict[str, Any]] = Field(default_factory=list)
    missing_in_json: List[Dict[str, Any]] = Field(default_factory=list)
    duplicates: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationRequest(BaseModel):
    gstr2b_file_id: str
    purchase_register_file_id: str
    similarity_threshold: Optional[float] = 0.85
    use_ml: Optional[bool] = False


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_size: int
    status: str = "success"


class ErrorResponse(BaseModel):
    message: str
    error_code: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)