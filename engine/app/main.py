import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.endpoints import upload, reconcile, reports
from app.config import settings

app = FastAPI(
    title="GST Invoice Reconciliation API",
    description="API for reconciling GSTR-2B data with purchase register",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(reconcile.router, prefix="/api/v1", tags=["Reconciliation"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to GST Invoice Reconciliation API"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)